"""Checks the bot's own logic without calling any model, search service or Metaculus.

Every model call is replaced by a stand-in that answers from a script, and the
step that turns a written answer into numbers is replaced by a plain reader of
the answer format the prompts ask for. Run with:

    uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

# Placeholders so nothing complains about missing keys. Nothing here reaches the network.
os.environ["METACULUS_TOKEN"] = "placeholder-for-tests"
os.environ["OPENROUTER_API_KEY"] = "placeholder-for-tests"
for variable in ("ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "BOT_FORECASTER_MODELS", "BOT_RESEARCH_MODELS", "BOT_HELPER_MODEL", "BOT_FORECASTS_PER_MODEL", "BOT_LOWEST_PROBABILITY"):
    os.environ.pop(variable, None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
import bot as bot_module  # noqa: E402
from forecasting_tools import (  # noqa: E402
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption  # noqa: E402

SONNET, SOL, ARGON = bot_module.DEFAULT_FORECASTERS


class Script:
    """What the stand-in models answer, and a record of every call made."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.binary = {SONNET: "20", SOL: "30", ARGON: "70"}
        self.failing: set[str] = set()
        self.failing_research: set[tuple[str, str]] = set()

    def kind_of(self, prompt: str) -> str:
        if "You are the research assistant of a professional forecaster" in prompt:
            return "research"
        if "Please summarize the following research" in prompt:
            return "summary"
        if 'Probability: ZZ%' in prompt:
            return "binary"
        if "Option_A: Probability_A" in prompt:
            return "choice"
        if "Percentile 95: XX" in prompt:
            return "numeric"
        return "other"

    async def invoke(self, llm: GeneralLlm, prompt, system_prompt=None) -> str:
        kind = self.kind_of(str(prompt))
        search = (llm.litellm_kwargs.get("extra_body") or {}).get("plugins", [{}])[0].get("engine", "any") if kind == "research" else ""
        self.calls.append((kind, llm.model, {"search": search, "prompt": str(prompt)}))
        if kind == "research":
            if (llm.model, search) in self.failing_research:
                raise RuntimeError("search unavailable")
            return f"The latest official figure is 4.1 percent (found by {llm.model})."
        if kind == "summary":
            return "A short summary."
        if llm.model in self.failing:
            raise RuntimeError("provider is down")
        if kind == "binary":
            return f"(a) Two months. (b) No. ... Probability: {self.binary[llm.model]}%"
        if kind == "choice":
            return "Reasoning.\nRed: 50%\nGreen: 30%\nBlue: 20%"
        if kind == "numeric":
            return "Reasoning.\nPercentile 5: 10\nPercentile 10: 20\nPercentile 20: 30\nPercentile 40: 45\nPercentile 60: 55\nPercentile 80: 70\nPercentile 90: 80\nPercentile 95: 90"
        return "ok"

    def models_for(self, kind: str) -> list[str]:
        return [model for called, model, _ in self.calls if called == kind]


async def read_answer(text_to_structure: str, output_type, **_ignored):
    """Stands in for the model that turns a written answer into numbers."""
    if output_type is BinaryPrediction:
        return BinaryPrediction(prediction_in_decimal=float(re.search(r"Probability: ([\d.]+)%", text_to_structure).group(1)) / 100)
    if output_type is PredictedOptionList:
        options = re.findall(r"^(\w+): (\d+)%$", text_to_structure, flags=re.MULTILINE)
        return PredictedOptionList(predicted_options=[PredictedOption(option_name=name, probability=int(share) / 100) for name, share in options])
    percentiles = re.findall(r"Percentile (\d+): ([\d.]+)", text_to_structure)
    return [Percentile(percentile=int(level) / 100, value=float(value)) for level, value in percentiles]


def binary_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the unemployment rate reported for November 2026 be above 4.0 percent?",
        resolution_criteria="Resolves Yes if the first official release shows a rate above 4.0 percent.",
        fine_print="Later revisions do not count.",
        background_info="The rate has been between 3.9 and 4.2 percent for a year.",
        page_url="https://www.metaculus.com/questions/1/",
        id_of_post=1,
        id_of_question=1,
    )


class BotTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.script = Script()
        script = self.script

        async def invoke(llm, prompt, system_prompt=None):
            return await script.invoke(llm, prompt, system_prompt)

        patches = [
            mock.patch.object(GeneralLlm, "invoke", invoke),
            mock.patch.object(main, "structure_output", read_answer),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def make_bot(self, **settings) -> bot_module.VentureBot:
        return bot_module.VentureBot(settings=bot_module.Settings(**settings), publish_reports_to_metaculus=False)

    async def test_each_model_forecasts_in_turn_and_the_middle_answer_is_used(self) -> None:
        report = await self.make_bot().forecast_question(binary_question())
        forecasters = self.script.models_for("binary")
        self.assertEqual(len(forecasters), 6)
        self.assertEqual({model: forecasters.count(model) for model in set(forecasters)}, {SONNET: 2, SOL: 2, ARGON: 2})
        # 20, 20, 30, 30, 70, 70: the median is 30.
        self.assertAlmostEqual(report.prediction, 0.30)
        self.assertEqual(report.errors, [])

    async def test_forecast_prompt_carries_the_question_the_research_and_the_date(self) -> None:
        await self.make_bot().forecast_question(binary_question())
        prompt = next(details["prompt"] for kind, _, details in self.script.calls if kind == "binary")
        self.assertIn("Will the unemployment rate reported for November 2026 be above 4.0 percent?", prompt)
        self.assertIn("Later revisions do not count.", prompt)
        self.assertIn("The latest official figure is 4.1 percent", prompt)
        self.assertRegex(prompt, r"Today is \d{4}-\d{2}-\d{2}\.")
        self.assertIn("(c) The base rate", prompt)
        # The research assistant is told not to forecast.
        research = next(details["prompt"] for kind, _, details in self.script.calls if kind == "research")
        self.assertIn("Give no probability and no prediction of your own.", research)

    async def test_extreme_answers_are_pulled_in(self) -> None:
        self.script.binary = {SONNET: "99.9", SOL: "99.9", ARGON: "99.9"}
        report = await self.make_bot().forecast_question(binary_question())
        self.assertAlmostEqual(report.prediction, 0.98)
        self.script.binary = {SONNET: "0.1", SOL: "0.1", ARGON: "0.1"}
        report = await self.make_bot(lowest_probability=0.05).forecast_question(binary_question())
        self.assertAlmostEqual(report.prediction, 0.05)

    async def test_one_provider_being_down_does_not_stop_the_forecast(self) -> None:
        self.script.failing = {ARGON}
        report = await self.make_bot().forecast_question(binary_question())
        # Four of six forecasts arrived: 20, 20, 30, 30.
        self.assertAlmostEqual(report.prediction, 0.25)
        self.assertEqual(len(report.errors), 2)

    async def test_too_many_failures_give_no_forecast_rather_than_a_thin_one(self) -> None:
        self.script.failing = {ARGON, SOL}
        with self.assertRaises(Exception):
            await self.make_bot().forecast_question(binary_question())

    async def test_web_research_asks_for_the_providers_own_search_first_then_falls_back(self) -> None:
        first, second = bot_module.DEFAULT_RESEARCHERS
        research = await self.make_bot().run_research(binary_question())
        self.assertIn("## Web research (by openai/gpt-6.1-terra", research)
        self.assertEqual([(model, details["search"]) for kind, model, details in self.script.calls if kind == "research"], [(first, "native")])

        self.script.calls.clear()
        self.script.failing_research = {(first, "native"), (first, "any")}
        research = await self.make_bot().run_research(binary_question())
        self.assertIn(f"found by {second}", research)
        self.assertEqual([(model, details["search"]) for kind, model, details in self.script.calls if kind == "research"], [(first, "native"), (first, "any"), (second, "native")])

    async def test_news_is_added_when_a_key_is_present_and_its_failure_is_survived(self) -> None:
        with mock.patch.dict(os.environ, {"ASKNEWS_API_KEY": "placeholder"}):
            with mock.patch.object(AskNewsSearcher, "__init__", lambda self, *a, **k: None), mock.patch.object(AskNewsSearcher, "call_preconfigured_version", mock.AsyncMock(return_value="Article: jobs report due December 4.")) as news:
                research = await self.make_bot().run_research(binary_question())
                news.assert_awaited_once_with("asknews/news-summaries", binary_question().question_text)
            self.assertIn("## Recent news articles\nArticle: jobs report due December 4.", research)
            self.assertLess(research.index("## Web research"), research.index("## Recent news articles"))

            with mock.patch.object(AskNewsSearcher, "__init__", lambda self, *a, **k: None), mock.patch.object(AskNewsSearcher, "call_preconfigured_version", mock.AsyncMock(side_effect=RuntimeError("quota"))):
                research = await self.make_bot().run_research(binary_question())
            self.assertIn("## Web research", research)
            self.assertNotIn("Recent news", research)

    async def test_with_no_research_at_all_the_forecaster_is_told_so(self) -> None:
        self.script.failing_research = {(model, search) for model in bot_module.DEFAULT_RESEARCHERS for search in ("native", "any")}
        research = await self.make_bot().run_research(binary_question())
        self.assertEqual(research, "No research could be gathered. Rely on what you know, and be less confident for it.")

    async def test_multiple_choice(self) -> None:
        question = MultipleChoiceQuestion(question_text="Which colour will be chosen?", options=["Red", "Green", "Blue"], page_url="https://www.metaculus.com/questions/2/", id_of_post=2, id_of_question=2)
        report = await self.make_bot().forecast_question(question)
        shares = {option.option_name: round(option.probability, 2) for option in report.prediction.predicted_options}
        self.assertEqual(shares, {"Red": 0.5, "Green": 0.3, "Blue": 0.2})
        self.assertEqual(len(self.script.models_for("choice")), 6)

    async def test_numeric(self) -> None:
        question = NumericQuestion(
            question_text="What will the index be on December 31?",
            upper_bound=100,
            lower_bound=0,
            open_upper_bound=True,
            open_lower_bound=False,
            unit_of_measure="points",
            page_url="https://www.metaculus.com/questions/3/",
            id_of_post=3,
            id_of_question=3,
        )
        report = await self.make_bot().forecast_question(question)
        # Six identical answers are combined, so the combined answer has the same shape as each.
        # Metaculus requires a little chance everywhere on the scale, which moves the points slightly.
        declared = {round(point.percentile, 2): point.value for point in report.prediction.declared_percentiles}
        self.assertAlmostEqual(declared[0.05], 10, delta=5)
        self.assertAlmostEqual(declared[0.95], 90, delta=5)
        self.assertLess(declared[0.05], declared[0.95])
        self.assertEqual(len(self.script.models_for("numeric")), 6)
        cdf = report.prediction.cdf
        self.assertEqual(len(cdf), 201)
        heights = [point.percentile for point in cdf]
        self.assertEqual(heights, sorted(heights))
        # The lower bound is closed, so no chance is placed below it; the upper one is open, so some is left above.
        self.assertAlmostEqual(heights[0], 0.0)
        self.assertLess(heights[-1], 1.0)

    def test_settings_come_from_the_environment_and_bad_values_fall_back(self) -> None:
        with mock.patch.dict(os.environ, {"BOT_FORECASTER_MODELS": " a/one , b/two ,", "BOT_FORECASTS_PER_MODEL": "3", "BOT_LOWEST_PROBABILITY": "nonsense", "BOT_RESEARCH_MODELS": ""}):
            settings = bot_module.Settings()
        self.assertEqual(settings.forecasters, ("a/one", "b/two"))
        self.assertEqual(settings.forecasts_per_model, 3)
        self.assertEqual(settings.lowest_probability, 0.02)
        self.assertEqual(settings.researchers, bot_module.DEFAULT_RESEARCHERS)
        with mock.patch.dict(os.environ, {"BOT_FORECASTS_PER_MODEL": "50", "BOT_LOWEST_PROBABILITY": "0.9"}):
            settings = bot_module.Settings()
        self.assertEqual(settings.forecasts_per_model, 5)
        self.assertEqual(settings.lowest_probability, 0.2)

    def test_the_bot_records_which_models_it_uses(self) -> None:
        config = self.make_bot().get_config()
        self.assertEqual(config["llms"]["default"]["model"], SONNET)
        self.assertEqual(config["llms"]["forecaster_2"]["model"], SOL)
        self.assertEqual(config["llms"]["forecaster_3"]["model"], ARGON)
        self.assertEqual(sorted(config["llms"]), ["default", "forecaster_2", "forecaster_3", "parser", "researcher", "summarizer"])
        self.assertEqual(config["predictions_per_research_report"], 6)
        self.assertEqual(config["settings"]["lowest_probability"], 0.02)

    def test_too_many_forecasting_models_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.make_bot(forecasters=tuple(f"m/{n}" for n in range(7)))


if __name__ == "__main__":
    unittest.main()
