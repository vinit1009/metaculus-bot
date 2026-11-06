"""Forecasts one made-up question with the real models and prints the result. Submits nothing.

A quick way to check that the configured models answer, that web research works and
that the answers parse, before pointing the bot at a tournament. Costs a few cents.

    uv run python scripts/forecast_one.py
    uv run python scripts/forecast_one.py --kind numeric
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The bot never talks to Metaculus here, but the framework expects a token to exist.
os.environ.setdefault("METACULUS_TOKEN", "not-used-by-this-script")

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion  # noqa: E402

import bot as bot_module  # noqa: E402


def sample(kind: str):
    resolves = (datetime.now(timezone.utc) + timedelta(days=45)).replace(microsecond=0)
    common = dict(page_url="https://example.invalid/sample", id_of_post=0, id_of_question=0, scheduled_resolution_time=resolves)
    if kind == "binary":
        return BinaryQuestion(
            question_text="Will the US unemployment rate for the most recent month reported before the resolution date be at or above 4.5%?",
            resolution_criteria="Resolves Yes if the first official Bureau of Labor Statistics release of the unemployment rate for the latest reported month is 4.5% or higher.",
            fine_print="Later revisions do not count.",
            background_info="The unemployment rate is published monthly by the Bureau of Labor Statistics in the Employment Situation report.",
            **common,
        )
    if kind == "numeric":
        return NumericQuestion(
            question_text="What will the US unemployment rate be in the most recent monthly report published before the resolution date?",
            resolution_criteria="Resolves to the first official Bureau of Labor Statistics figure for the latest reported month, in percent.",
            fine_print="Later revisions do not count.",
            background_info="The unemployment rate is published monthly by the Bureau of Labor Statistics.",
            unit_of_measure="percent",
            upper_bound=8.0,
            lower_bound=2.0,
            open_upper_bound=True,
            open_lower_bound=True,
            **common,
        )
    return MultipleChoiceQuestion(
        question_text="In the most recent monthly report published before the resolution date, which range will the US unemployment rate fall in?",
        options=["Below 4.0%", "4.0% to 4.4%", "4.5% to 4.9%", "5.0% or above"],
        resolution_criteria="Resolves to the range containing the first official Bureau of Labor Statistics figure for the latest reported month.",
        background_info="The unemployment rate is published monthly by the Bureau of Labor Statistics.",
        **common,
    )


async def main(kind: str) -> int:
    forecasting_bot = bot_module.VentureBot(publish_reports_to_metaculus=False)
    print("Models:", ", ".join(forecasting_bot.settings.forecasters))
    print("Research:", ", ".join(forecasting_bot.settings.researchers), "| helper:", forecasting_bot.settings.helper)
    report = await forecasting_bot.forecast_question(sample(kind))
    print("\n=== Result")
    print("Prediction:", report.make_readable_prediction(report.prediction))
    print(f"Cost: ${report.price_estimate:.4f}   Minutes: {report.minutes_taken:.1f}   Errors: {len(report.errors)}")
    for error in report.errors:
        print("  error:", str(error)[:300])
    print("\n=== Explanation (first 3000 characters)")
    print(report.explanation[:3000])
    return 0 if not report.errors else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Forecast one made-up question with the real models")
    parser.add_argument("--kind", choices=["binary", "numeric", "choice"], default="binary")
    sys.exit(asyncio.run(main(parser.parse_args().kind)))
