"""A forecasting bot for the Metaculus FutureEval tournament and MiniBench.

It is built on the official template in main.py, which is left untouched so
that updates to the template can be merged. What this file adds:

- Several models, in turn. Each question is forecast a few times by each of
  several current models from different providers. The framework then combines
  the answers (the median, for yes/no questions). Models trained by different
  companies make different mistakes, so their middle is steadier than any one.
- Two kinds of research, side by side. A news search (AskNews, when a key is
  present) and a web search carried out by a model that is asked for the
  current status, what is scheduled before the deadline, base rates, and what
  markets or official projections say.
- Base rates and the exact resolution rules are made explicit steps in the
  reasoning before an answer is given.
- Narrower limits on extreme answers. Being wrong at 99% costs far more under
  the tournament's scoring than is gained by being right at 99% instead of 98%.

Nothing here is tuned by hand to particular questions, and no person adjusts a
forecast: the bot reads a question, researches it, and submits.

Settings can be changed without touching the code, through environment
variables (see `Settings`).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from main import FallTemplateBot2026  # also loads .env and quiets noisy libraries

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
)
from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    DateQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
)

logger = logging.getLogger(__name__)

# Model names as listed by OpenRouter on 2026-11-06. One model from each of the
# three companies whose credits the tournament donates.
DEFAULT_FORECASTERS = (
    "openrouter/anthropic/claude-sonnet-5.5",
    "openrouter/openai/gpt-6.1-sol",
    "openrouter/google/gemini-4-argon",
)
# Tried in order until one answers. Each is asked to search the web itself.
DEFAULT_RESEARCHERS = (
    "openrouter/openai/gpt-6.1-terra",
    "openrouter/google/gemini-3.8-flash",
)
# Turns a written answer into numbers, and writes the short research summary.
DEFAULT_HELPER = "openrouter/openai/gpt-6-luna"

# The most models that can take turns forecasting.
MAX_FORECASTERS = 6

# The provider's own web search, switched on through OpenRouter.
NATIVE_WEB_SEARCH = {"plugins": [{"id": "web", "engine": "native"}]}
# OpenRouter's choice of search engine, for a model with none of its own.
ANY_WEB_SEARCH = {"plugins": [{"id": "web"}]}


def _names(variable: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """A comma-separated list from the environment, or the default when unset or empty."""
    names = tuple(name.strip() for name in os.getenv(variable, "").split(",") if name.strip())
    return names or default


def _number(variable: str, default: float, lowest: float, highest: float) -> float:
    """A number from the environment, kept within sensible limits. A bad value falls back to the default."""
    try:
        value = float(os.getenv(variable, "") or default)
    except ValueError:
        logger.warning(f"{variable} is not a number; using {default}")
        return default
    return min(max(value, lowest), highest)


@dataclass(frozen=True)
class Settings:
    """What can be changed from outside. Each field names its environment variable."""

    # BOT_FORECASTER_MODELS: the models that forecast, in turn.
    forecasters: tuple[str, ...] = field(default_factory=lambda: _names("BOT_FORECASTER_MODELS", DEFAULT_FORECASTERS))
    # BOT_RESEARCH_MODELS: models that can search the web, tried in order.
    researchers: tuple[str, ...] = field(default_factory=lambda: _names("BOT_RESEARCH_MODELS", DEFAULT_RESEARCHERS))
    # BOT_HELPER_MODEL: a small model for parsing and summarising.
    helper: str = field(default_factory=lambda: _names("BOT_HELPER_MODEL", (DEFAULT_HELPER,))[0])
    # BOT_FORECASTS_PER_MODEL: how many times each model forecasts a question.
    forecasts_per_model: int = field(default_factory=lambda: int(_number("BOT_FORECASTS_PER_MODEL", 2, 1, 5)))
    # BOT_LOWEST_PROBABILITY: no yes/no answer goes below this, or above one minus this.
    lowest_probability: float = field(default_factory=lambda: _number("BOT_LOWEST_PROBABILITY", 0.02, 0.001, 0.2))


def _forecaster(model: str) -> GeneralLlm:
    # These models reason before answering and do not take a temperature. They
    # can think for minutes on a hard question, so the time limit is generous.
    return GeneralLlm(model=model, temperature=None, timeout=300, allowed_tries=2)


def _researcher(model: str, search: dict) -> GeneralLlm:
    return GeneralLlm(model=model, temperature=None, timeout=240, allowed_tries=1, extra_body=search)


def _has_asknews() -> bool:
    return bool(os.getenv("ASKNEWS_API_KEY") or (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _timing(question: MetaculusQuestion) -> str:
    """A sentence on when the question closes and resolves, when Metaculus says."""
    parts = []
    if question.close_time:
        parts.append(f"Forecasting on it closes on {question.close_time.strftime('%Y-%m-%d')}.")
    if question.scheduled_resolution_time:
        parts.append(f"It is scheduled to resolve by {question.scheduled_resolution_time.strftime('%Y-%m-%d')}.")
    return " ".join(parts)


class VentureBot(FallTemplateBot2026):
    """The template bot with several forecasting models, combined research and firmer limits."""

    def __init__(self, *, settings: Settings | None = None, **kwargs) -> None:
        self.settings = settings or Settings()
        if len(self.settings.forecasters) > MAX_FORECASTERS:
            raise ValueError(f"At most {MAX_FORECASTERS} forecasting models can be used, but {len(self.settings.forecasters)} were given.")
        self._forecasters = [_forecaster(model) for model in self.settings.forecasters]
        self._turn = itertools.count()
        helper = GeneralLlm(model=self.settings.helper, temperature=None, timeout=90, allowed_tries=3)
        llms: dict[str, str | GeneralLlm | None] = {
            # "default" is what the framework records as the forecaster. The others take their turn through get_llm.
            "default": self._forecasters[0],
            "summarizer": helper,
            "parser": helper,
            # Research is done in run_research below, not by a single model named here.
            "researcher": "no_research",
        }
        for position in range(2, MAX_FORECASTERS + 1):
            llms[f"forecaster_{position}"] = self._forecasters[position - 1] if position <= len(self._forecasters) else None
        kwargs.setdefault("research_reports_per_question", 1)
        kwargs.setdefault("predictions_per_research_report", self.settings.forecasts_per_model * len(self._forecasters))
        kwargs.setdefault("use_research_summary_to_forecast", False)
        kwargs.setdefault("extra_metadata_in_explanation", True)
        super().__init__(llms=llms, **kwargs)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        # The framework checks every model purpose against this list. Every purpose is
        # set explicitly in __init__, so none of these defaults is ever used. The
        # template's own defaults name models from 2024, which is why they are not kept.
        defaults: dict[str, str | GeneralLlm | None] = {"default": None, "summarizer": None, "researcher": None, "parser": None}
        for position in range(2, MAX_FORECASTERS + 1):
            defaults[f"forecaster_{position}"] = None
        return defaults

    def get_config(self) -> dict[str, Any]:
        # Recorded with every forecast, so it is clear afterwards how each one was made.
        config = super().get_config()
        config["settings"] = asdict(self.settings)
        config["research_reports_per_question"] = self.research_reports_per_question
        config["predictions_per_research_report"] = self.predictions_per_research_report
        config["use_research_summary_to_forecast"] = self.use_research_summary_to_forecast
        config["llms"] = {purpose: llm for purpose, llm in config["llms"].items() if llm is not None}
        return config

    def get_llm(self, purpose: str = "default", guarantee_type: Literal["llm", "string_name"] | None = None):  # type: ignore[override]
        # The template asks for the "default" model once per forecast. Handing
        # out the forecasters in turn spreads the forecasts evenly over them.
        if purpose == "default" and guarantee_type == "llm":
            return self._forecasters[next(self._turn) % len(self._forecasters)]
        return super().get_llm(purpose, guarantee_type)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ research

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            news, web = await asyncio.gather(self._news(question), self._web_research(question))
            research = "\n\n".join(part for part in (web, news) if part)
            if not research:
                logger.warning(f"No research could be gathered for {question.page_url}")
                research = "No research could be gathered. Rely on what you know, and be less confident for it."
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _news(self, question: MetaculusQuestion) -> str:
        if not _has_asknews():
            return ""
        try:
            found = await AskNewsSearcher().call_preconfigured_version("asknews/news-summaries", question.question_text)
        except Exception as error:  # a search that fails must not stop the forecast
            logger.warning(f"News search failed for {question.page_url}: {error.__class__.__name__}: {error}")
            return ""
        return f"## Recent news articles\n{found}" if found and found.strip() else ""

    async def _web_research(self, question: MetaculusQuestion) -> str:
        prompt = self._web_research_prompt(question)
        for model in self.settings.researchers:
            for search in (NATIVE_WEB_SEARCH, ANY_WEB_SEARCH):
                try:
                    found = await _researcher(model, search).invoke(prompt)
                except Exception as error:
                    logger.warning(f"Web research with {model} failed for {question.page_url}: {error.__class__.__name__}: {error}")
                    continue
                if found and found.strip():
                    return f"## Web research (by {model.removeprefix('openrouter/')}, {_today()})\n{found}"
        return ""

    @staticmethod
    def _web_research_prompt(question: MetaculusQuestion) -> str:
        return clean_indents(
            f"""
            You are the research assistant of a professional forecaster. Search the web and report what bears on the question below.
            You do not forecast. Give no probability and no prediction of your own.

            Question:
            {question.question_text}

            How it will be resolved:
            {question.resolution_criteria}

            {question.fine_print}

            Background given with the question:
            {question.background_info}

            Today is {_today()}. {_timing(question)}

            Report the following, each with dates and where it comes from:
            1. Where things stand now: the latest facts that bear directly on how this resolves. If the outcome already looks decided, say so plainly and say why.
            2. What is scheduled between now and the resolution date: votes, rulings, releases, deadlines, meetings, data publications.
            3. Base rates: how often this kind of thing has happened before in comparable periods, with the numbers.
            4. If the question asks for a number or a date: the latest value, its recent range and direction, and the source that will be used to resolve the question.
            5. What prediction markets, bookmakers, official projections or published forecasts say, with their numbers, if any exist.
            6. The strongest reasons the outcome could differ from what now looks most likely.

            Prefer primary and recent sources. Where you find nothing, say that you found nothing instead of guessing.
            Keep the report under 700 words.
            """
        )

    # -------------------------------------------------------------- yes/no questions

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {_today()}. {_timing(question)}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) The base rate: how often events like this have happened in comparable situations, and the probability that alone suggests.
            (d) What the resolution criteria and fine print require exactly, and whether that makes Yes or No easier to reach than the headline suggests.
            (e) A brief description of a scenario that results in a No outcome.
            (f) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time. They start from the base rate and move away from it only as far as the specific evidence justifies, and they are no more confident than that evidence allows.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        forecast = await self._binary_prompt_to_forecast(question, prompt)
        low = self.settings.lowest_probability
        return ReasonedPrediction(prediction_value=min(max(forecast.prediction_value, low), 1 - low), reasoning=forecast.reasoning)

    # ------------------------------------------------------ multiple choice questions

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {_today()}. {_timing(question)}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) The base rate: how often each kind of outcome has come about in comparable situations.
            (d) What the resolution criteria and fine print require exactly, and which option each plausible outcome would fall under.
            (e) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    # ------------------------------------------------------------ numeric questions

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {_today()}. {_timing(question)}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The latest known value, and the outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) How much values like this have moved over a period of this length in the past.
            (e) The expectations of experts and markets.
            (f) A brief description of an unexpected scenario that results in a low outcome.
            (g) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns, and that the 5th and 95th percentiles lie well outside the range they think likely.

            The last thing you write is your final answer as:
            "
            Percentile 5: XX (lowest number value)
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            Percentile 95: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    # Date questions use the template's own prompt unchanged.
    async def _run_forecast_on_date(self, question: DateQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        return await super()._run_forecast_on_date(question, research)


TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def run(run_mode: Literal["tournament", "metaculus_cup", "test_questions"], publish: bool = True) -> None:
    check_environment(strict=True)
    print_startup_banner(run_mode, will_publish=publish)
    bot = VentureBot(
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )
    client = MetaculusClient()
    if run_mode == "tournament":
        # The season's main tournament and MiniBench. A question already forecast is skipped.
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        reports += asyncio.run(bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True))
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True))
    else:
        # The bot testing area holds every kind of question, for checking a new setup.
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
    bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=TOURNAMENT_URLS.get(run_mode))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run the forecasting bot")
    parser.add_argument("--mode", type=str, choices=["tournament", "metaculus_cup", "test_questions"], default="tournament", help="What to forecast on (default: tournament)")
    parser.add_argument("--dry-run", action="store_true", help="Forecast but do not submit anything to Metaculus")
    arguments = parser.parse_args()
    run(arguments.mode, publish=not arguments.dry_run)
