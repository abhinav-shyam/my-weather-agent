from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


WeatherDetail = Literal[
    "temperature",
    "apparent_temperature",
    "precipitation",
    "wind",
    "conditions",
]


class WeatherIntent(BaseModel):
    locations: list[str] = Field(default_factory=list)
    intent: Literal["current", "forecast", "historical", "non_weather"]
    forecast_granularity: Literal["daily", "hourly"] = "daily"
    time_of_day: Literal["all_day", "morning", "afternoon", "evening", "night"] = (
        "all_day"
    )
    requested_details: list[WeatherDetail] = Field(
        default_factory=lambda: [
            "temperature",
            "apparent_temperature",
            "precipitation",
            "wind",
            "conditions",
        ]
    )
    start_date: str
    end_date: str
    clarification_needed: bool
    clarification_question: str
