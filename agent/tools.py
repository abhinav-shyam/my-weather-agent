from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

import httpx
from langchain_core.tools import tool

from utils.time_utils import today_str

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_CUTOFF_DAYS = 7
HTTP_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
API_MAX_RETRIES = 3
API_RETRY_BACKOFF_SECONDS = 1.0
API_CONCURRENCY_LIMIT = 5
_API_SEMAPHORE = asyncio.Semaphore(API_CONCURRENCY_LIMIT)

CURRENT_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "weathercode",
    "windspeed_10m",
    "windgusts_10m",
    "winddirection_10m",
    "is_day",
]

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "weathercode",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "winddirection_10m_dominant",
    "uv_index_max",
    "snowfall_sum",
]

HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "weathercode",
    "windspeed_10m",
    "windgusts_10m",
    "winddirection_10m",
    "relative_humidity_2m",
]

TIME_OF_DAY_HOURS = {
    "all_day": None,
    "morning": set(range(6, 12)),
    "afternoon": set(range(12, 17)),
    "evening": set(range(17, 21)),
    "night": set(range(0, 6)) | set(range(21, 24)),
}


async def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send a GET request to Open-Meteo and return a validated JSON object."""
    last_error: Exception | None = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            async with _API_SEMAPHORE:
                async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                    response = await client.get(url, params=params)

            if response.status_code == 429:
                raise RuntimeError("Open-Meteo rate limit reached (HTTP 429).")

            if response.status_code >= 500:
                raise RuntimeError(
                    f"Open-Meteo temporary server error ({response.status_code})."
                )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Open-Meteo request failed with status {response.status_code}: {response.text}"
                )

            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Open-Meteo response was not a JSON object.")
            return payload
        except (httpx.TimeoutException, httpx.NetworkError, RuntimeError) as exc:
            last_error = exc
            if attempt >= API_MAX_RETRIES or not _is_retryable_error(exc):
                raise
            await asyncio.sleep(API_RETRY_BACKOFF_SECONDS * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Open-Meteo request failed unexpectedly.")


def _is_retryable_error(error: Exception) -> bool:
    """Return whether a request failure is transient enough to retry."""
    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    message = str(error).lower()
    return "429" in message or "temporary server error" in message


def _validate_geocode_payload(
    payload: dict[str, Any], location_name: str
) -> dict[str, Any]:
    """Normalize the best geocoding match and ensure required coordinates exist."""
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"Location not found: {location_name}")

    best_match = results[0]
    required_fields = ("latitude", "longitude")
    missing_fields = [field for field in required_fields if field not in best_match]
    if missing_fields:
        raise ValueError(
            f"Geocoding response missing required fields: {', '.join(missing_fields)}"
        )

    return {
        "latitude": best_match["latitude"],
        "longitude": best_match["longitude"],
        "timezone": best_match.get("timezone", "auto"),
        "name": best_match.get("name", location_name),
        "country": best_match.get("country"),
    }


def _validate_weather_payload(
    payload: dict[str, Any], intent: str, forecast_granularity: str = "daily"
) -> dict[str, Any]:
    """Check that the weather payload contains the section required for the request."""
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))

    required_top_level = {"latitude", "longitude", "timezone"}
    missing_top_level = [field for field in required_top_level if field not in payload]
    if missing_top_level:
        raise ValueError(
            f"Weather response missing required fields: {', '.join(missing_top_level)}"
        )

    if intent == "current":
        current = payload.get("current")
        if not isinstance(current, dict):
            raise ValueError("Weather response missing 'current' data.")
    elif intent == "forecast" and forecast_granularity == "hourly":
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise ValueError("Weather response missing 'hourly' data.")
        times = hourly.get("time")
        if not isinstance(times, list) or not times:
            raise ValueError("Weather response missing hourly time series.")
    else:
        daily = payload.get("daily")
        if not isinstance(daily, dict):
            raise ValueError("Weather response missing 'daily' data.")
        times = daily.get("time")
        if not isinstance(times, list) or not times:
            raise ValueError("Weather response missing daily time series.")

    return payload


def _slice_daily_payload(
    payload: dict[str, Any], start_date: str, end_date: str
) -> dict[str, Any]:
    """Trim daily time-series fields to the inclusive requested date range."""
    daily = payload.get("daily")
    if not isinstance(daily, dict):
        return payload

    times = daily.get("time")
    if not isinstance(times, list):
        return payload

    selected_indexes = [
        index
        for index, value in enumerate(times)
        if isinstance(value, str) and start_date <= value <= end_date
    ]
    if not selected_indexes:
        raise ValueError(
            f"Weather response did not include data for requested range {start_date} to {end_date}."
        )

    sliced_daily: dict[str, Any] = {}
    for key, values in daily.items():
        if isinstance(values, list):
            sliced_daily[key] = [
                values[index] for index in selected_indexes if index < len(values)
            ]
        else:
            sliced_daily[key] = values

    updated_payload = dict(payload)
    updated_payload["daily"] = sliced_daily
    return updated_payload


def _parse_hourly_timestamp(value: str) -> datetime:
    """Parse an ISO hourly timestamp returned by Open-Meteo."""
    return datetime.fromisoformat(value)


def _slice_hourly_payload(
    payload: dict[str, Any],
    start_date: str,
    end_date: str,
    time_of_day: str,
) -> dict[str, Any]:
    """Trim hourly series to the requested dates and optional time-of-day window."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        return payload

    times = hourly.get("time")
    if not isinstance(times, list):
        return payload

    allowed_hours = TIME_OF_DAY_HOURS.get(time_of_day)
    selected_indexes: list[int] = []
    for index, value in enumerate(times):
        if not isinstance(value, str):
            continue
        timestamp = _parse_hourly_timestamp(value)
        date_value = timestamp.date().isoformat()
        if not (start_date <= date_value <= end_date):
            continue
        if allowed_hours is not None and timestamp.hour not in allowed_hours:
            continue
        selected_indexes.append(index)

    if not selected_indexes:
        raise ValueError(
            "Weather response did not include hourly data for the requested range "
            f"{start_date} to {end_date} ({time_of_day})."
        )

    sliced_hourly: dict[str, Any] = {}
    for key, values in hourly.items():
        if isinstance(values, list):
            sliced_hourly[key] = [
                values[index] for index in selected_indexes if index < len(values)
            ]
        else:
            sliced_hourly[key] = values

    updated_payload = dict(payload)
    updated_payload["hourly"] = sliced_hourly
    return updated_payload


@tool
async def geocode(location_name: str) -> dict[str, Any]:
    """Resolve a place name into latitude, longitude, and timezone metadata."""
    params = {
        "name": location_name,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    payload = await _request_json(GEOCODING_URL, params)
    return _validate_geocode_payload(payload, location_name)


def _parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD string into a date object."""
    return datetime.strptime(value, "%Y-%m-%d").date()


@tool
async def fetch_weather(
    latitude: float,
    longitude: float,
    intent: str,
    forecast_granularity: str,
    time_of_day: str,
    start_date: str,
    end_date: str,
    timezone: str,
) -> dict[str, Any]:
    """Fetch current, forecast, or historical weather for a location and date range."""
    common_params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone or "auto",
        "temperature_unit": "celsius",
    }

    today = _parse_date(today_str())
    start = _parse_date(start_date)
    end = _parse_date(end_date)

    if intent == "current":
        params = {
            **common_params,
            "current": ",".join(CURRENT_VARS),
        }
        payload = await _request_json(FORECAST_URL, params)
        return _validate_weather_payload(payload, intent)

    if intent == "forecast":
        params = {
            **common_params,
            "start_date": start_date,
            "end_date": end_date,
        }
        if forecast_granularity == "hourly":
            params["hourly"] = ",".join(HOURLY_VARS)
        else:
            params["daily"] = ",".join(DAILY_VARS)
        payload = await _request_json(FORECAST_URL, params)
        payload = _validate_weather_payload(payload, intent, forecast_granularity)
        if forecast_granularity == "hourly":
            payload = _slice_hourly_payload(payload, start_date, end_date, time_of_day)
        payload["_requested_range"] = {
            "start_date": start_date,
            "end_date": end_date,
        }
        payload["_forecast_granularity"] = forecast_granularity
        payload["_time_of_day"] = time_of_day
        return payload

    if intent != "historical":
        raise ValueError(f"Unsupported intent: {intent}")

    age_in_days = (today - start).days
    if age_in_days > ARCHIVE_CUTOFF_DAYS:
        params = {
            **common_params,
            "daily": ",".join(DAILY_VARS),
            "start_date": start_date,
            "end_date": end_date,
        }
        payload = await _request_json(ARCHIVE_URL, params)
        return _validate_weather_payload(payload, intent)

    past_days = max(1, age_in_days + 1)
    params = {
        **common_params,
        "daily": ",".join(DAILY_VARS),
        "past_days": past_days,
    }
    payload = await _request_json(FORECAST_URL, params)
    payload["_requested_range"] = {
        "start_date": start_date,
        "end_date": end_date,
    }
    payload = _validate_weather_payload(payload, intent)
    return _slice_daily_payload(payload, start_date, end_date)


WEATHER_TOOLS = [geocode, fetch_weather]
