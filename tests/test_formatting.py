"""Tests for data formatting and helper functions."""

from __future__ import annotations

import pytest

from agent.nodes import (
    DEFAULT_REQUESTED_DETAILS,
    _build_hourly_points,
    _format_current_response,
    _format_hourly_response,
    _format_multi_location_response,
    _is_broad_summary,
    _is_greeting,
    _location_label,
    _normalize_locations,
    _requested_details,
    _sample_indexes,
    _weather_code_description,
    _wind_phrase,
)


class TestNormalizeLocations:
    """Test location normalization (core extraction logic)."""

    def test_deduplication_case_insensitive(self) -> None:
        """Test that locations are deduplicated case-insensitively."""
        parsed = {"locations": ["Toronto", "toronto", "TORONTO", "Vancouver", "vancouver"]}
        result = _normalize_locations(parsed)
        assert len(result) == 2
        assert "Toronto" in result
        assert "Vancouver" in result

    def test_preserves_first_occurrence_case(self) -> None:
        """Test that first occurrence's case is preserved."""
        parsed = {"locations": ["New York", "new york"]}
        result = _normalize_locations(parsed)
        assert result[0] == "New York"

    def test_whitespace_normalization(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        parsed = {"locations": ["  Toronto  ", " Vancouver ", "\tMontreal\n"]}
        result = _normalize_locations(parsed)
        assert all(loc == loc.strip() for loc in result)

    def test_filters_non_string_values(self) -> None:
        """Test that non-string values are filtered out."""
        parsed = {"locations": ["Toronto", 123, None, "Vancouver", [], {}]}
        result = _normalize_locations(parsed)
        assert result == ["Toronto", "Vancouver"]


class TestIsGreeting:
    """Test greeting detection for non-weather queries."""

    def test_simple_greetings(self) -> None:
        """Test that basic greetings are detected."""
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "yo"]
        for greeting in greetings:
            assert _is_greeting(greeting) is True

    def test_greetings_with_context(self) -> None:
        """Test that greetings are detected even with surrounding text."""
        assert _is_greeting("hello how are you") is True
        assert _is_greeting("hey there") is True

    def test_case_insensitive_matching(self) -> None:
        """Test that greeting detection is case-insensitive."""
        assert _is_greeting("HELLO") is True
        assert _is_greeting("Hello") is True
        assert _is_greeting("hElLo") is True

    def test_non_greetings_rejected(self) -> None:
        """Test that non-greetings are not detected."""
        non_greetings = ["weather in Toronto", "temperature", "tell me something", "help me"]
        for phrase in non_greetings:
            assert _is_greeting(phrase) is False


class TestSampleIndexes:
    """Test index sampling for reducing large datasets."""

    def test_even_distribution(self) -> None:
        """Test that samples are evenly distributed."""
        result = _sample_indexes(100, max_points=5)
        assert len(result) == 5
        assert result[0] == 0
        assert result[-1] == 99
        
        # Check roughly even spacing
        gaps = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        assert all(20 <= gap <= 35 for gap in gaps)

    def test_no_duplicates(self) -> None:
        """Test that sampled indexes have no duplicates."""
        result = _sample_indexes(1000, max_points=10)
        assert len(result) == len(set(result))

    def test_sorted_output(self) -> None:
        """Test that samples are returned in sorted order."""
        result = _sample_indexes(500, max_points=8)
        assert result == sorted(result)

    def test_respects_max_points_limit(self) -> None:
        """Test that output never exceeds max_points."""
        for max_pts in [3, 5, 10, 20]:
            result = _sample_indexes(1000, max_points=max_pts)
            assert len(result) <= max_pts

    def test_small_dataset_returns_all(self) -> None:
        """Test that datasets smaller than max_points return all indexes."""
        result = _sample_indexes(3, max_points=10)
        assert result == [0, 1, 2]


class TestWeatherCodeDescription:
    """Test WMO weather code interpretation."""

    def test_standard_weather_codes(self) -> None:
        """Test that standard WMO codes are interpreted correctly."""
        code_tests = [
            (0, "clear skies"),
            (3, "overcast skies"),
            (61, "light rain"),
            (73, "snow"),
            (95, "thunderstorms"),
        ]
        for code, expected in code_tests:
            assert _weather_code_description(code) == expected

    def test_unknown_code_fallback(self) -> None:
        """Test that unknown codes return sensible default."""
        assert _weather_code_description(999) == "mixed conditions"
        assert _weather_code_description(-1) == "mixed conditions"

    def test_invalid_types_handled(self) -> None:
        """Test that invalid types don't crash."""
        assert _weather_code_description("invalid") == "mixed conditions"
        assert _weather_code_description(None) == "mixed conditions"
        assert _weather_code_description({}) == "mixed conditions"


class TestWindPhrase:
    """Test wind speed description formatting."""

    def test_wind_speed_categories(self) -> None:
        """Test that wind speeds are categorized correctly."""
        # Light wind (< 10 km/h)
        assert "light wind" in _wind_phrase(3.0)
        assert "3.0 km/h" in _wind_phrase(3.0)

        # Steady wind (10-20 km/h)
        assert "steady wind" in _wind_phrase(15.0)

        # Brisk wind (> 20 km/h)
        assert "brisk wind" in _wind_phrase(30.0)

    def test_zero_wind(self) -> None:
        """Test that zero wind is described as light."""
        phrase = _wind_phrase(0.0)
        assert "light wind" in phrase

    def test_invalid_inputs_return_empty(self) -> None:
        """Test that invalid inputs return empty string."""
        assert _wind_phrase("invalid") == ""
        assert _wind_phrase(None) == ""
        assert _wind_phrase([]) == ""


class TestRequestedDetails:
    """Test extraction of requested weather details."""

    def test_explicit_details_returned(self) -> None:
        """Test that explicitly set details are returned."""
        weather_data = {"_requested_details": ["temperature", "wind"]}
        result = _requested_details(weather_data)
        assert result == ["temperature", "wind"]

    def test_missing_details_uses_defaults(self) -> None:
        """Test that missing details fall back to defaults."""
        weather_data = {}
        result = _requested_details(weather_data)
        assert result == DEFAULT_REQUESTED_DETAILS

    def test_invalid_format_uses_defaults(self) -> None:
        """Test that invalid format falls back to defaults."""
        weather_data = {"_requested_details": "not a list"}
        result = _requested_details(weather_data)
        assert result == DEFAULT_REQUESTED_DETAILS


class TestIsBroadSummary:
    """Test detection of broad (all details) vs focused requests."""

    def test_all_default_details_is_broad(self) -> None:
        """Test that all default details indicates broad summary."""
        assert _is_broad_summary(DEFAULT_REQUESTED_DETAILS) is True

    def test_reordered_defaults_still_broad(self) -> None:
        """Test that order doesn't matter for broad detection."""
        reordered = list(reversed(DEFAULT_REQUESTED_DETAILS))
        assert _is_broad_summary(reordered) is True

    def test_subset_not_broad(self) -> None:
        """Test that subsets are not considered broad."""
        assert _is_broad_summary(["temperature"]) is False
        assert _is_broad_summary(["temperature", "wind"]) is False

    def test_superset_not_broad(self) -> None:
        """Test that extra details make it non-broad."""
        extra = list(DEFAULT_REQUESTED_DETAILS) + ["unknown_detail"]
        assert _is_broad_summary(extra) is False


class TestLocationLabel:
    """Test location name extraction from weather data."""

    def test_extracts_location_name(self) -> None:
        """Test that location name is correctly extracted."""
        weather_data = {"_location": {"name": "Toronto"}}
        assert _location_label(weather_data) == "Toronto"

    def test_fallback_to_raw_location(self) -> None:
        """Test fallback when _location dict is missing name."""
        weather_data = {"_location": {}, "_raw_location": "Toronto"}
        assert _location_label(weather_data) == "Toronto"

    def test_default_when_missing(self) -> None:
        """Test default is used when no location info."""
        assert _location_label({}) == "that location"
        assert _location_label({"_location": "invalid"}) == "that location"


class TestFormatCurrentResponse:
    """Test current weather response formatting."""

    def test_respects_requested_details(self, weather_data_fixture) -> None:
        """Test that response only includes requested details."""
        weather_data_fixture["_requested_details"] = ["temperature"]
        response = _format_current_response(weather_data_fixture)
        
        assert "22.5" in response  # Temperature is included
        assert response  # Non-empty

    def test_includes_location_name(self, weather_data_fixture) -> None:
        """Test that location name is included in response."""
        response = _format_current_response(weather_data_fixture)
        assert "Toronto" in response

    def test_handles_missing_current_data(self) -> None:
        """Test that missing current data returns empty."""
        weather_data = {"_requested_details": ["temperature"]}
        response = _format_current_response(weather_data)
        assert response == ""

    def test_broad_summary_includes_all_details(self, weather_data_fixture) -> None:
        """Test that broad summary includes all available details."""
        response = _format_current_response(weather_data_fixture)
        assert len(response) > 0


class TestFormatHourlyResponse:
    """Test hourly forecast response formatting."""

    def test_hourly_includes_location(self, hourly_data_fixture) -> None:
        """Test that hourly response includes location name."""
        response = _format_hourly_response(hourly_data_fixture)
        assert "Paris" in response

    def test_hourly_includes_time_data(self, hourly_data_fixture) -> None:
        """Test that hourly response contains time information."""
        response = _format_hourly_response(hourly_data_fixture)
        # Should have hour labels
        assert any(str(h) in response for h in range(0, 24))

    def test_empty_hourly_data_returns_empty(self) -> None:
        """Test that missing hourly data returns empty."""
        weather_data = {"hourly": {}}
        response = _format_hourly_response(weather_data)
        assert response == ""


class TestBuildHourlyPoints:
    """Test hourly data point building."""

    def test_builds_valid_point_structure(self, hourly_data_fixture) -> None:
        """Test that hourly points have required fields."""
        points = _build_hourly_points(hourly_data_fixture)
        assert len(points) > 0
        
        point = points[0]
        required_fields = ["label", "temperature"]
        assert all(field in point for field in required_fields)

    def test_respects_max_point_limit(self, hourly_data_fixture) -> None:
        """Test that output is limited to max points (default 5)."""
        points = _build_hourly_points(hourly_data_fixture)
        assert len(points) <= 5

    def test_empty_hourly_returns_empty_list(self) -> None:
        """Test that missing hourly data returns empty."""
        weather_data = {"hourly": {}, "_requested_details": ["temperature"]}
        points = _build_hourly_points(weather_data)
        assert points == []


class TestMultiLocationResponse:
    """Test multi-location response formatting."""

    def test_formats_multiple_locations(self, weather_data_fixture, forecast_data_fixture) -> None:
        """Test that multiple locations are formatted correctly."""
        locations = [weather_data_fixture, forecast_data_fixture]
        
        def simple_formatter(data):
            loc_name = _location_label(data)
            return f"Weather for {loc_name}"
        
        result = _format_multi_location_response(locations, simple_formatter)
        assert "Toronto" in result
        assert "London" in result
        assert "\n\n" in result  # Locations are separated

    def test_includes_errors_in_response(self, weather_data_fixture) -> None:
        """Test that location errors are included."""
        locations = [
            weather_data_fixture,
            {"_raw_location": "InvalidCity", "error": "Location not found"},
        ]
        
        def formatter(data):
            return "Weather"
        
        result = _format_multi_location_response(locations, formatter)
        assert "InvalidCity" in result
        assert "Location not found" in result
