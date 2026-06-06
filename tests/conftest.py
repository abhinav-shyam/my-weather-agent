"""Shared fixtures for weather agent tests."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.nodes import AgentState


@pytest.fixture
def basic_state() -> AgentState:
    """Provide a minimal valid AgentState for testing."""
    return {
        "messages": [HumanMessage(content="What is the weather in New York?")],
        "parsed": {},
        "weather_data": {},
        "tool_trace": [],
        "response": "",
    }


@pytest.fixture
def parsed_state() -> AgentState:
    """Provide an AgentState with parsed intent."""
    return {
        "messages": [HumanMessage(content="Current weather in Toronto")],
        "parsed": {
            "locations": ["Toronto"],
            "intent": "current",
            "forecast_granularity": "daily",
            "time_of_day": "all_day",
            "requested_details": ["temperature", "conditions"],
            "start_date": "2026-06-06",
            "end_date": "2026-06-06",
            "clarification_needed": False,
            "clarification_question": "",
        },
        "weather_data": {},
        "tool_trace": [],
        "response": "",
    }


@pytest.fixture
def weather_data_fixture() -> dict:
    """Provide mock weather data for current conditions."""
    return {
        "_location": {
            "latitude": 43.65,
            "longitude": -79.38,
            "timezone": "America/Toronto",
            "name": "Toronto",
            "country": "Canada",
        },
        "_raw_location": "Toronto",
        "_requested_details": ["temperature", "apparent_temperature", "conditions"],
        "current": {
            "temperature_2m": 22.5,
            "apparent_temperature": 21.0,
            "relative_humidity_2m": 65,
            "precipitation": 0.0,
            "weathercode": 2,
            "windspeed_10m": 12.5,
            "windgusts_10m": 18.0,
            "winddirection_10m": 200,
            "is_day": True,
        },
    }


@pytest.fixture
def forecast_data_fixture() -> dict:
    """Provide mock weather data for daily forecast."""
    return {
        "_location": {
            "latitude": 51.51,
            "longitude": -0.13,
            "timezone": "Europe/London",
            "name": "London",
            "country": "United Kingdom",
        },
        "_raw_location": "London",
        "_requested_details": ["temperature", "precipitation"],
        "daily": {
            "time": ["2026-06-06", "2026-06-07", "2026-06-08"],
            "temperature_2m_max": [19.5, 20.0, 18.5],
            "temperature_2m_min": [14.2, 15.0, 13.8],
            "precipitation_sum": [0.0, 2.5, 1.2],
            "precipitation_probability_max": [10, 65, 45],
            "weathercode": [2, 3, 3],
        },
    }


@pytest.fixture
def hourly_data_fixture() -> dict:
    """Provide mock weather data for hourly forecast."""
    return {
        "_location": {
            "latitude": 48.85,
            "longitude": 2.35,
            "timezone": "Europe/Paris",
            "name": "Paris",
            "country": "France",
        },
        "_raw_location": "Paris",
        "_requested_details": ["temperature", "precipitation"],
        "_time_of_day": "all_day",
        "hourly": {
            "time": [
                f"2026-06-06T{h:02d}:00" for h in range(24)
            ],
            "temperature_2m": list(range(15, 28)) + list(range(27, 14, -1)),
            "apparent_temperature": list(range(14, 27)) + list(range(26, 13, -1)),
            "precipitation_probability": [0] * 8 + [10] * 4 + [20] * 4 + [15] * 8,
            "precipitation": [0.0] * 12 + [0.5] * 8 + [0.2] * 4,
            "weathercode": [0] * 8 + [2] * 10 + [3] * 6,
            "windspeed_10m": [5.0 + i * 0.5 for i in range(24)],
        },
    }


@pytest.fixture
def non_weather_state() -> AgentState:
    """Provide an AgentState with non-weather intent."""
    return {
        "messages": [HumanMessage(content="Hello!")],
        "parsed": {
            "intent": "non_weather",
            "locations": [],
            "clarification_needed": False,
        },
        "weather_data": {},
        "tool_trace": [],
        "response": "",
    }


@pytest.fixture
def greeting_state() -> AgentState:
    """Provide an AgentState with a greeting message."""
    return {
        "messages": [HumanMessage(content="Hi there")],
        "parsed": {},
        "weather_data": {},
        "tool_trace": [],
        "response": "",
    }
