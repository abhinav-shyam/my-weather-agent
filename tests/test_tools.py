"""Tests for weather tools."""

from __future__ import annotations

import pytest

from agent.tools import (
    _is_retryable_error,
    _validate_geocode_payload,
    _validate_weather_payload,
)


class TestRetryableError:
    """Test error retry logic."""

    def test_timeout_is_retryable(self) -> None:
        import httpx
        error = httpx.TimeoutException("timeout")
        assert _is_retryable_error(error) is True

    def test_network_error_is_retryable(self) -> None:
        import httpx
        error = httpx.NetworkError("connection failed")
        assert _is_retryable_error(error) is True

    def test_rate_limit_is_retryable(self) -> None:
        error = RuntimeError("429 rate limit")
        assert _is_retryable_error(error) is True

    def test_server_error_is_retryable(self) -> None:
        error = RuntimeError("temporary server error (503)")
        assert _is_retryable_error(error) is True

    def test_invalid_error_not_retryable(self) -> None:
        error = RuntimeError("Invalid request")
        assert _is_retryable_error(error) is False


class TestValidateGeocodePayload:
    """Test geocoding response validation."""

    def test_valid_geocode_response(self) -> None:
        payload = {
            "results": [
                {
                    "latitude": 43.65,
                    "longitude": -79.38,
                    "timezone": "America/Toronto",
                    "name": "Toronto",
                    "country": "Canada",
                }
            ]
        }
        
        result = _validate_geocode_payload(payload, "Toronto")
        assert result["latitude"] == 43.65
        assert result["longitude"] == -79.38
        assert result["name"] == "Toronto"

    def test_missing_latitude(self) -> None:
        payload = {
            "results": [
                {
                    "longitude": -79.38,
                    "timezone": "America/Toronto",
                    "name": "Toronto",
                }
            ]
        }
        
        with pytest.raises(ValueError, match="missing required fields"):
            _validate_geocode_payload(payload, "Toronto")

    def test_missing_longitude(self) -> None:
        payload = {
            "results": [
                {
                    "latitude": 43.65,
                    "timezone": "America/Toronto",
                    "name": "Toronto",
                }
            ]
        }
        
        with pytest.raises(ValueError, match="missing required fields"):
            _validate_geocode_payload(payload, "Toronto")

    def test_empty_results(self) -> None:
        payload = {"results": []}
        
        with pytest.raises(ValueError, match="Location not found"):
            _validate_geocode_payload(payload, "InvalidCity")

    def test_no_results_key(self) -> None:
        payload: dict = {}
        
        with pytest.raises(ValueError, match="Location not found"):
            _validate_geocode_payload(payload, "Toronto")

    def test_default_timezone(self) -> None:
        payload = {
            "results": [
                {
                    "latitude": 43.65,
                    "longitude": -79.38,
                    "name": "Toronto",
                }
            ]
        }
        
        result = _validate_geocode_payload(payload, "Toronto")
        assert result["timezone"] == "auto"

    def test_uses_first_result(self) -> None:
        payload = {
            "results": [
                {
                    "latitude": 43.65,
                    "longitude": -79.38,
                    "timezone": "America/Toronto",
                    "name": "Toronto, ON",
                },
                {
                    "latitude": 47.65,
                    "longitude": -79.38,
                    "timezone": "America/Toronto",
                    "name": "Toronto, Canada",
                },
            ]
        }
        
        result = _validate_geocode_payload(payload, "Toronto")
        assert result["name"] == "Toronto, ON"


class TestValidateWeatherPayload:
    """Test weather response validation."""

    def test_valid_current_weather(self) -> None:
        payload = {
            "latitude": 43.65,
            "longitude": -79.38,
            "timezone": "America/Toronto",
            "current": {
                "temperature_2m": 22.5,
                "weathercode": 2,
            }
        }
        
        result = _validate_weather_payload(payload, "current")
        assert result["latitude"] == 43.65

    def test_missing_current_data(self) -> None:
        payload = {
            "latitude": 43.65,
            "longitude": -79.38,
            "timezone": "America/Toronto",
        }
        
        with pytest.raises(ValueError, match="missing 'current' data"):
            _validate_weather_payload(payload, "current")

    def test_valid_daily_forecast(self) -> None:
        payload = {
            "latitude": 43.65,
            "longitude": -79.38,
            "timezone": "America/Toronto",
            "daily": {
                "time": ["2026-06-06", "2026-06-07"],
                "temperature_2m_max": [25.0, 24.0],
            }
        }
        
        result = _validate_weather_payload(payload, "forecast", "daily")
        assert result["latitude"] == 43.65

    def test_missing_daily_data(self) -> None:
        payload = {
            "latitude": 43.65,
            "longitude": -79.38,
            "timezone": "America/Toronto",
        }
        
        with pytest.raises(ValueError, match="missing 'daily' data"):
            _validate_weather_payload(payload, "forecast", "daily")

    def test_empty_daily_times(self) -> None:
        payload = {
            "latitude": 43.65,
            "longitude": -79.38,
            "timezone": "America/Toronto",
            "daily": {"time": []},
        }
        
        with pytest.raises(ValueError, match="missing daily time series"):
            _validate_weather_payload(payload, "forecast", "daily")

    def test_valid_hourly_forecast(self) -> None:
        payload = {
            "latitude": 43.65,
            "longitude": -79.38,
            "timezone": "America/Toronto",
            "hourly": {
                "time": ["2026-06-06T00:00", "2026-06-06T01:00"],
                "temperature_2m": [20.0, 21.0],
            }
        }
        
        result = _validate_weather_payload(payload, "forecast", "hourly")
        assert result["latitude"] == 43.65

    def test_missing_hourly_data(self) -> None:
        payload = {
            "latitude": 43.65,
            "longitude": -79.38,
            "timezone": "America/Toronto",
        }
        
        with pytest.raises(ValueError, match="missing 'hourly' data"):
            _validate_weather_payload(payload, "forecast", "hourly")

    def test_error_in_payload(self) -> None:
        payload = {
            "error": "API error occurred"
        }
        
        with pytest.raises(RuntimeError, match="API error occurred"):
            _validate_weather_payload(payload, "current")

    def test_missing_required_fields(self) -> None:
        payload = {
            "latitude": 43.65,
            "timezone": "America/Toronto",
            "current": {"temperature_2m": 22.5},
        }
        
        with pytest.raises(ValueError, match="missing required fields"):
            _validate_weather_payload(payload, "current")
