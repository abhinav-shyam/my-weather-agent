from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.llm import call_llm, call_structured_llm
from agent.prompts import (
    build_error_response_prompt,
    build_parse_prompt,
    build_parse_system_prompt,
    build_response_prompt,
    build_response_system_prompt,
)
from agent.state import AgentState
from agent.tools import fetch_weather, geocode
from utils.time_utils import current_datetime_str

GREETING_KEYWORDS = {
    "hello",
    "hi",
    "hey",
    "hola",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "sup",
    "whats up",
    "yo",
}

TIME_OF_DAY_LABELS = {
    "all_day": "Today",
    "morning": "This morning",
    "afternoon": "This afternoon",
    "evening": "This evening",
    "night": "Tonight",
}

WMO_CODE_DESCRIPTIONS = {
    0: "clear skies",
    1: "mostly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "foggy conditions",
    48: "icy fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "severe thunderstorms with hail",
}

DEFAULT_REQUESTED_DETAILS = [
    "temperature",
    "apparent_temperature",
    "precipitation",
    "wind",
    "conditions",
]


def _normalize_locations(parsed: dict[str, Any]) -> list[str]:
    """Return a de-duplicated list of parsed locations in first-seen order."""
    candidates = parsed.get("locations", [])
    if not isinstance(candidates, list):
        candidates = []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, str):
            continue
        location = value.strip()
        key = location.casefold()
        if not location or key in seen:
            continue
        seen.add(key)
        normalized.append(location)
    return normalized


def _last_user_message(messages: list[BaseMessage]) -> str:
    """Return the most recent human-authored message content from the chat history."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _is_greeting(text: str) -> bool:
    """Check whether the user input looks like a greeting rather than a weather request."""
    normalized = " ".join(text.lower().split())
    return any(
        normalized == keyword or normalized.startswith(f"{keyword} ")
        for keyword in GREETING_KEYWORDS
    )


def _format_hour_label(value: str) -> str:
    dt = datetime.fromisoformat(value)
    return dt.strftime("%I %p").lstrip("0")


def _sample_indexes(length: int, max_points: int = 5) -> list[int]:
    if length <= max_points:
        return list(range(length))

    step = (length - 1) / (max_points - 1)
    return sorted({round(step * i) for i in range(max_points)})


def _weather_code_description(value: Any) -> str:
    if isinstance(value, int):
        return WMO_CODE_DESCRIPTIONS.get(value, "mixed conditions")
    return "mixed conditions"


def _wind_phrase(speed: Any) -> str:
    if not isinstance(speed, (int, float)):
        return ""
    if speed < 5:
        return f"light wind near **{speed:.1f} km/h**"
    if speed < 15:
        return f"gentle wind near **{speed:.1f} km/h**"
    if speed < 25:
        return f"steady wind near **{speed:.1f} km/h**"
    return f"brisk wind near **{speed:.1f} km/h**"


def _requested_details(weather_data: dict[str, Any]) -> list[str]:
    details = weather_data.get("_requested_details")
    if isinstance(details, list):
        return [str(item) for item in details]
    return list(DEFAULT_REQUESTED_DETAILS)


def _is_broad_summary(details: list[str]) -> bool:
    return set(details) == set(DEFAULT_REQUESTED_DETAILS)


def _build_hourly_points(weather_data: dict[str, Any]) -> list[dict[str, Any]]:
    hourly = weather_data.get("hourly")
    if not isinstance(hourly, dict):
        return []

    times = hourly.get("time")
    if not isinstance(times, list):
        return []

    points: list[dict[str, Any]] = []
    for index in _sample_indexes(len(times)):
        raw_time = times[index]
        if not isinstance(raw_time, str):
            continue

        point: dict[str, Any] = {
            "label": _format_hour_label(raw_time),
            "temperature": None,
            "apparent_temperature": None,
            "precipitation_probability": None,
            "weathercode": None,
            "windspeed": None,
        }

        mapping = {
            "temperature_2m": "temperature",
            "apparent_temperature": "apparent_temperature",
            "precipitation_probability": "precipitation_probability",
            "weathercode": "weathercode",
            "windspeed_10m": "windspeed",
        }
        for source_key, target_key in mapping.items():
            values = hourly.get(source_key)
            if isinstance(values, list) and index < len(values):
                point[target_key] = values[index]

        points.append(point)

    return points


def _hourly_summary(weather_data: dict[str, Any], points: list[dict[str, Any]]) -> str:
    location = weather_data.get("_location", {})
    if not isinstance(location, dict):
        location = {}
    location_name = str(location.get("name", "that location"))
    time_of_day = str(weather_data.get("_time_of_day", "all_day"))
    label = TIME_OF_DAY_LABELS.get(time_of_day, "Today")
    condition = (
        _weather_code_description(points[0].get("weathercode"))
        if points
        else "mixed conditions"
    )
    details = _requested_details(weather_data)

    if details == ["temperature"]:
        return f"{label} in {location_name} temperatures:"
    if details == ["apparent_temperature"]:
        return f"{label} in {location_name} feels like:"
    if details == ["precipitation"]:
        return f"{label} in {location_name} rain chances:"
    if details == ["wind"]:
        return f"{label} in {location_name} winds:"
    if details == ["conditions"]:
        return f"{label} in {location_name} looks like {condition}."
    return f"{label} in {location_name} looks like {condition}."


def _hourly_recommendation(points: list[dict[str, Any]]) -> str:
    precip_values = [
        value
        for value in (point.get("precipitation_probability") for point in points)
        if isinstance(value, (int, float))
    ]
    temp_values = [
        value
        for value in (point.get("temperature") for point in points)
        if isinstance(value, (int, float))
    ]

    max_precip = max(precip_values, default=0)
    min_temp = min(temp_values, default=None)

    if max_precip >= 50:
        return "A small umbrella would be a good idea if you're heading out."
    if min_temp is not None and min_temp <= 18:
        return "A light layer should be enough if you're out for a while."
    return "Conditions look comfortable for being outside."


def _format_hourly_response(weather_data: dict[str, Any]) -> str:
    points = _build_hourly_points(weather_data)
    if not points:
        return ""

    details = _requested_details(weather_data)
    lines = [_hourly_summary(weather_data, points), ""]
    for point in points:
        parts: list[str] = []

        temperature = point.get("temperature")
        if "temperature" in details and isinstance(temperature, (int, float)):
            parts.append(f"**{temperature:.1f} C**")

        apparent = point.get("apparent_temperature")
        if "apparent_temperature" in details and isinstance(apparent, (int, float)):
            parts.append(f"feels like **{apparent:.1f} C**")

        condition = _weather_code_description(point.get("weathercode"))
        if "conditions" in details and condition:
            parts.append(condition)

        precip = point.get("precipitation_probability")
        if "precipitation" in details and isinstance(precip, (int, float)):
            parts.append(f"**{precip:.0f}%** rain chance")

        wind = _wind_phrase(point.get("windspeed"))
        if "wind" in details and wind:
            parts.append(wind)

        if not parts:
            continue
        lines.append(f"- **{point['label']}:** " + ", ".join(parts))

    if _is_broad_summary(details):
        lines.append("")
        lines.append(_hourly_recommendation(points))
    return "\n".join(lines)


def _current_summary(
    location_name: str, details: list[str], current: dict[str, Any]
) -> str:
    if details == ["temperature"]:
        value = current.get("temperature_2m")
        if isinstance(value, (int, float)):
            return f"It is **{value:.1f} C** in {location_name}."
    if details == ["apparent_temperature"]:
        value = current.get("apparent_temperature")
        if isinstance(value, (int, float)):
            return f"It feels like **{value:.1f} C** in {location_name}."
    if details == ["precipitation"]:
        amount = current.get("precipitation")
        if isinstance(amount, (int, float)):
            return f"Current precipitation in {location_name} is **{amount:.1f} mm**."
    if details == ["wind"]:
        speed = current.get("windspeed_10m")
        if isinstance(speed, (int, float)):
            return f"Wind in {location_name} is around **{speed:.1f} km/h**."
    return f"Current weather in {location_name}:"


def _format_current_response(weather_data: dict[str, Any]) -> str:
    current = weather_data.get("current")
    if not isinstance(current, dict):
        return ""

    location = weather_data.get("_location", {})
    if not isinstance(location, dict):
        location = {}
    location_name = str(location.get("name", "that location"))
    details = _requested_details(weather_data)

    if details == ["temperature"]:
        value = current.get("temperature_2m")
        if isinstance(value, (int, float)):
            return f"It is **{value:.1f} C** in {location_name}."
    if details == ["apparent_temperature"]:
        value = current.get("apparent_temperature")
        if isinstance(value, (int, float)):
            return f"It feels like **{value:.1f} C** in {location_name}."
    if details == ["precipitation"]:
        amount = current.get("precipitation")
        if isinstance(amount, (int, float)):
            return f"Current precipitation in {location_name} is **{amount:.1f} mm**."
    if details == ["wind"]:
        speed = current.get("windspeed_10m")
        if isinstance(speed, (int, float)):
            return f"Wind in {location_name} is around **{speed:.1f} km/h**."

    lines = [_current_summary(location_name, details, current), ""]

    if "temperature" in details:
        value = current.get("temperature_2m")
        if isinstance(value, (int, float)):
            lines.append(f"- Temperature: **{value:.1f} C**")

    if "apparent_temperature" in details:
        value = current.get("apparent_temperature")
        if isinstance(value, (int, float)):
            lines.append(f"- Feels like: **{value:.1f} C**")

    if "precipitation" in details:
        amount = current.get("precipitation")
        if isinstance(amount, (int, float)):
            lines.append(f"- Current precipitation: **{amount:.1f} mm**")

    if "wind" in details:
        speed = current.get("windspeed_10m")
        gusts = current.get("windgusts_10m")
        if isinstance(speed, (int, float)):
            wind_line = f"- Wind: **{speed:.1f} km/h**"
            if isinstance(gusts, (int, float)):
                wind_line += f", gusts up to **{gusts:.1f} km/h**"
            lines.append(wind_line)

    if "conditions" in details:
        lines.append(
            f"- Conditions: {_weather_code_description(current.get('weathercode'))}"
        )

    return "\n".join(line for line in lines if line)


def _build_parse_messages(state: AgentState) -> list[BaseMessage]:
    """Assemble the message list used to parse the user's weather intent."""
    history = list(state.get("messages", []))
    system = SystemMessage(content=build_parse_system_prompt(current_datetime_str()))
    user_request = _last_user_message(history)
    parse_prompt = HumanMessage(content=build_parse_prompt(user_request))
    return [system, *history, parse_prompt]


def _location_label(weather_data: dict[str, Any]) -> str:
    location = weather_data.get("_location", {})
    if isinstance(location, dict):
        name = location.get("name")
        if isinstance(name, str) and name.strip():
            return name
    raw_location = weather_data.get("_raw_location")
    if isinstance(raw_location, str) and raw_location.strip():
        return raw_location
    return "that location"


def _format_location_error(weather_data: dict[str, Any]) -> str:
    error = weather_data.get("error")
    if not isinstance(error, str) or not error:
        return ""
    location_name = _location_label(weather_data)
    return f"I couldn't fetch the weather for {location_name}: {error}"


def _format_multi_location_response(
    locations: list[dict[str, Any]], formatter: Any
) -> str:
    sections: list[str] = []
    for weather_data in locations:
        if not isinstance(weather_data, dict):
            continue
        if "error" in weather_data:
            error_text = _format_location_error(weather_data)
            if error_text:
                sections.append(error_text)
            continue
        rendered = formatter(weather_data)
        if rendered:
            sections.append(rendered)
    return "\n\n".join(sections)


async def _fetch_location_weather(
    location: str,
    parsed: dict[str, Any],
    requested_details: list[str],
) -> dict[str, Any]:
    """Fetch weather for a single parsed location and annotate the payload."""
    try:
        geo = await geocode.ainvoke({"location_name": location})
        weather_payload = await fetch_weather.ainvoke(
            {
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "intent": parsed["intent"],
                "forecast_granularity": parsed.get("forecast_granularity", "daily"),
                "time_of_day": parsed.get("time_of_day", "all_day"),
                "start_date": parsed["start_date"],
                "end_date": parsed["end_date"],
                "timezone": geo["timezone"],
            }
        )
        weather_payload["_location"] = geo
        weather_payload["_raw_location"] = location
        weather_payload["_requested_details"] = list(requested_details)
        return weather_payload
    except Exception as exc:
        return {
            "_location": {"name": location},
            "_raw_location": location,
            "_requested_details": list(requested_details),
            "error": str(exc),
        }


def _build_response_messages(state: AgentState) -> list[BaseMessage]:
    """Build the message list used to generate the final user-facing response."""
    history = list(state.get("messages", []))
    system = SystemMessage(content=build_response_system_prompt(current_datetime_str()))
    weather_data = state.get("weather_data", {})

    if "error" in weather_data:
        prompt = build_error_response_prompt(str(weather_data["error"]))
    else:
        prompt = build_response_prompt(weather_data)

    return [system, *history, HumanMessage(content=prompt)]


async def parse_intent(state: AgentState) -> AgentState:
    """Parse the latest user request into structured weather intent data."""
    prompt_messages = _build_parse_messages(state)
    try:
        result = await call_structured_llm(prompt_messages)
        parsed = result.model_dump()
    except Exception as exc:
        fallback_message = (
            "I ran into a problem understanding that request. Please try again "
            "with the city and time period you want."
        )
        tool_trace = list(state.get("tool_trace", []))
        tool_trace.append({"parse_error": str(exc)})
        return {
            **state,
            "parsed": {"clarification_needed": True},
            "messages": [AIMessage(content=fallback_message)],
            "tool_trace": tool_trace,
            "response": fallback_message,
        }

    tool_trace = list(state.get("tool_trace", []))
    tool_trace.append({"parse_output": parsed})

    parsed["locations"] = _normalize_locations(parsed)

    updated_state: AgentState = {
        **state,
        "parsed": parsed,
        "tool_trace": tool_trace,
    }

    if parsed.get("intent") == "non_weather":
        return updated_state

    if not parsed["locations"] and not parsed.get("clarification_needed"):
        clarification_question = "Which location would you like the weather for?"
        updated_state["parsed"]["clarification_needed"] = True
        updated_state["parsed"]["clarification_question"] = clarification_question
        updated_state["messages"] = [AIMessage(content=clarification_question)]
        updated_state["response"] = clarification_question
        return updated_state

    if parsed.get("clarification_needed"):
        clarification_question = parsed.get(
            "clarification_question",
            "Could you clarify the location and time period?",
        )
        updated_state["messages"] = [AIMessage(content=clarification_question)]
        updated_state["response"] = clarification_question

    return updated_state


async def fetch_weather_node(state: AgentState) -> AgentState:
    """Resolve the parsed locations and fetch weather data for the requested date range."""
    parsed = state.get("parsed", {})
    tool_trace = list(state.get("tool_trace", []))
    requested_details = parsed.get("requested_details", list(DEFAULT_REQUESTED_DETAILS))
    locations = _normalize_locations(parsed)

    try:
        weather_results = await asyncio.gather(
            *[
                _fetch_location_weather(location, parsed, requested_details)
                for location in locations
            ]
        )

        if not weather_results:
            raise ValueError("No locations were provided for the weather lookup.")

        tool_trace.append(
            {
                "tools": [geocode.name, fetch_weather.name],
                "locations": [_location_label(result) for result in weather_results],
                "intent": parsed["intent"],
                "forecast_granularity": parsed.get("forecast_granularity", "daily"),
                "requested_details": list(requested_details),
                "time_of_day": parsed.get("time_of_day", "all_day"),
                "start_date": parsed["start_date"],
                "end_date": parsed["end_date"],
            }
        )

        if len(weather_results) == 1:
            only_result = weather_results[0]
            if "error" in only_result:
                return {
                    **state,
                    "weather_data": {"error": str(only_result["error"])},
                    "tool_trace": tool_trace,
                }
            return {
                **state,
                "weather_data": only_result,
                "tool_trace": tool_trace,
            }

        weather_payload = {
            "locations": weather_results,
            "_intent": parsed["intent"],
            "_forecast_granularity": parsed.get("forecast_granularity", "daily"),
            "_time_of_day": parsed.get("time_of_day", "all_day"),
            "_requested_details": list(requested_details),
        }
        return {
            **state,
            "weather_data": weather_payload,
            "tool_trace": tool_trace,
        }
    except Exception as exc:
        tool_trace.append(
            {"tools": [geocode.name, fetch_weather.name], "error": str(exc)}
        )
        return {
            **state,
            "weather_data": {"error": str(exc)},
            "tool_trace": tool_trace,
        }


async def handle_non_weather(state: AgentState) -> AgentState:
    """Reply to greetings or non-weather requests with a redirect back to weather help."""
    parsed = state.get("parsed", {})
    tool_trace = list(state.get("tool_trace", []))
    tool_trace.append({"route": "non_weather"})

    last_user_message = _last_user_message(list(state.get("messages", [])))
    if _is_greeting(last_user_message):
        clarification = (
            "Hello! I am your weather assistant. Glad to assist you. "
            "Where would you like the weather for?"
        )
    else:
        clarification = parsed.get(
            "clarification_question",
            "Hello! I am your weather assistant. Glad to assist you. "
            "Where would you like the weather for?",
        )

    return {
        **state,
        "messages": [AIMessage(content=clarification)],
        "response": clarification,
        "tool_trace": tool_trace,
    }


async def generate_response(state: AgentState) -> AgentState:
    """Generate the final assistant reply from fetched weather data or an error payload."""
    weather_data = state.get("weather_data", {})
    multi_location_data = weather_data.get("locations")
    if isinstance(multi_location_data, list) and multi_location_data:
        if weather_data.get("_intent") == "current":
            final_response = _format_multi_location_response(
                multi_location_data, _format_current_response
            )
            if final_response:
                return {
                    **state,
                    "messages": [AIMessage(content=final_response)],
                    "response": final_response,
                }

        if weather_data.get("_forecast_granularity") == "hourly":
            final_response = _format_multi_location_response(
                multi_location_data, _format_hourly_response
            )
            if final_response:
                return {
                    **state,
                    "messages": [AIMessage(content=final_response)],
                    "response": final_response,
                }

    if (
        isinstance(weather_data.get("current"), dict)
        and weather_data.get("_forecast_granularity") != "hourly"
    ):
        final_response = _format_current_response(weather_data)
        if final_response:
            return {
                **state,
                "messages": [AIMessage(content=final_response)],
                "response": final_response,
            }

    if weather_data.get("_forecast_granularity") == "hourly":
        final_response = _format_hourly_response(weather_data)
        if final_response:
            return {
                **state,
                "messages": [AIMessage(content=final_response)],
                "response": final_response,
            }

    prompt_messages = _build_response_messages(state)

    try:
        final_response = await call_llm(prompt_messages)
    except Exception:
        final_response = "I wasn't able to complete the weather lookup right now. Please try again shortly."

    return {
        **state,
        "messages": [AIMessage(content=final_response)],
        "response": final_response,
    }
