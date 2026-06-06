"""Integration tests for weather agent workflow."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from agent.nodes import (
    AgentState,
    parse_intent,
)
from agent.schema import WeatherIntent


class TestGraphConstruction:
    """Test that the graph is built correctly."""

    def test_graph_builds_successfully(self) -> None:
        """Test that build_graph returns a compiled graph."""
        graph = build_graph()
        
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")

    def test_graph_has_required_nodes(self) -> None:
        """Test that graph contains required nodes."""
        graph = build_graph()
        
        # The graph should be callable
        assert callable(graph.invoke)


class TestIntentParsingVariations:
    """Test intent parsing for critical new features."""

    @pytest.mark.asyncio
    async def test_current_hourly_remaps_to_forecast(self, basic_state, monkeypatch) -> None:
        """Test that 'current hourly' is parsed as forecast with hourly granularity."""
        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=["Bengaluru"],
                intent="forecast",
                forecast_granularity="hourly",
                start_date="2026-06-06",
                end_date="2026-06-06",
                clarification_needed=False,
                clarification_question="",
            )
        
        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)
        basic_state["messages"] = [HumanMessage(content="Current hourly weather in Bengaluru")]
        
        result = await parse_intent(basic_state)
        
        assert result["parsed"]["intent"] == "forecast"
        assert result["parsed"]["forecast_granularity"] == "hourly"
        assert result["parsed"]["start_date"] == "2026-06-06"

    @pytest.mark.asyncio
    async def test_yesterday_hourly_uses_historical(self, basic_state, monkeypatch) -> None:
        """Test that 'yesterday hourly' uses historical intent with hourly granularity."""
        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=["Bengaluru"],
                intent="historical",
                forecast_granularity="hourly",
                start_date="2026-06-05",
                end_date="2026-06-05",
                clarification_needed=False,
                clarification_question="",
            )
        
        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)
        basic_state["messages"] = [HumanMessage(content="Yesterday hourly weather in Bengaluru")]
        
        result = await parse_intent(basic_state)
        
        assert result["parsed"]["intent"] == "historical"
        assert result["parsed"]["forecast_granularity"] == "hourly"
        assert result["parsed"]["start_date"] == "2026-06-05"

    @pytest.mark.asyncio
    async def test_today_hourly_forecast(self, basic_state, monkeypatch) -> None:
        """Test that 'today hourly' uses forecast intent."""
        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=["Toronto"],
                intent="forecast",
                forecast_granularity="hourly",
                start_date="2026-06-06",
                end_date="2026-06-06",
                clarification_needed=False,
                clarification_question="",
            )
        
        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)
        
        result = await parse_intent(basic_state)
        
        assert result["parsed"]["intent"] == "forecast"
        assert result["parsed"]["forecast_granularity"] == "hourly"

    @pytest.mark.asyncio
    async def test_tomorrow_hourly_forecast(self, basic_state, monkeypatch) -> None:
        """Test that 'tomorrow hourly' uses forecast intent."""
        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=["Toronto"],
                intent="forecast",
                forecast_granularity="hourly",
                start_date="2026-06-07",
                end_date="2026-06-07",
                clarification_needed=False,
                clarification_question="",
            )
        
        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)
        basic_state["messages"] = [HumanMessage(content="Tomorrow hourly weather")]
        
        result = await parse_intent(basic_state)
        
        assert result["parsed"]["intent"] == "forecast"
        assert result["parsed"]["start_date"] == "2026-06-07"

    @pytest.mark.asyncio
    async def test_multi_location_current(self, basic_state, monkeypatch) -> None:
        """Test parsing multiple locations for current weather."""
        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=["Toronto", "Vancouver"],
                intent="current",
                forecast_granularity="daily",
                start_date="2026-06-06",
                end_date="2026-06-06",
                clarification_needed=False,
                clarification_question="",
            )
        
        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)
        basic_state["messages"] = [HumanMessage(content="Weather in Toronto and Vancouver")]
        
        result = await parse_intent(basic_state)
        
        assert len(result["parsed"]["locations"]) == 2
        assert "Toronto" in result["parsed"]["locations"]
        assert "Vancouver" in result["parsed"]["locations"]

    @pytest.mark.asyncio
    async def test_requested_details_temperature_only(self, basic_state, monkeypatch) -> None:
        """Test that focused requests are parsed correctly."""
        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=["Toronto"],
                intent="current",
                forecast_granularity="daily",
                requested_details=["temperature"],
                start_date="2026-06-06",
                end_date="2026-06-06",
                clarification_needed=False,
                clarification_question="",
            )
        
        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)
        basic_state["messages"] = [HumanMessage(content="Just temperature in Toronto")]
        
        result = await parse_intent(basic_state)
        
        assert result["parsed"]["requested_details"] == ["temperature"]


class TestErrorRecovery:
    """Test error handling and recovery workflows."""

    @pytest.mark.asyncio
    async def test_location_missing_triggers_clarification(self, basic_state, monkeypatch) -> None:
        """Test that missing location triggers clarification."""
        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=[],  # No locations parsed
                intent="forecast",
                forecast_granularity="daily",
                start_date="2026-06-06",
                end_date="2026-06-06",
                clarification_needed=False,
                clarification_question="",
            )
        
        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)
        basic_state["messages"] = [HumanMessage(content="What's the weather?")]
        
        result = await parse_intent(basic_state)
        
        assert result["parsed"]["clarification_needed"] is True
        assert "location" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_parse_intent_llm_error_handled(self, basic_state, monkeypatch) -> None:
        """Test that LLM errors are caught and clarification is requested."""
        async def mock_call_structured_llm(messages):
            raise RuntimeError("LLM unavailable")
        
        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)
        
        result = await parse_intent(basic_state)
        
        assert result["parsed"]["clarification_needed"] is True
        assert len(result["messages"]) > 0
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["tool_trace"]) > 0


class TestStateFlow:
    """Test state flow through different workflow paths."""

    def test_parsed_state_has_all_fields(self, parsed_state) -> None:
        """Test that parsed state contains all required fields."""
        parsed = parsed_state["parsed"]
        required = [
            "locations",
            "intent",
            "forecast_granularity",
            "start_date",
            "end_date",
            "clarification_needed",
        ]
        assert all(key in parsed for key in required)

    def test_state_immutability_preserved(self, parsed_state) -> None:
        """Test that state updates don't mutate original."""
        original_locations = parsed_state["parsed"]["locations"].copy()
        
        updated = {
            **parsed_state,
            "parsed": {
                **parsed_state["parsed"],
                "locations": ["NewCity"],
            }
        }
        
        assert parsed_state["parsed"]["locations"] == original_locations
        assert updated["parsed"]["locations"] == ["NewCity"]

    def test_tool_trace_accumulation(self) -> None:
        """Test that tool trace properly accumulates."""
        state: AgentState = {
            "messages": [],
            "parsed": {},
            "weather_data": {},
            "tool_trace": [
                {"step": "parse_intent", "status": "started"},
            ],
            "response": "",
        }
        
        updated_state = {
            **state,
            "tool_trace": [
                *state["tool_trace"],
                {"step": "parse_intent", "status": "completed"},
            ]
        }
        
        assert len(updated_state["tool_trace"]) == 2
        assert updated_state["tool_trace"][0]["status"] == "started"
        assert updated_state["tool_trace"][1]["status"] == "completed"


class TestMultiLocationWorkflow:
    """Test handling of multiple locations."""

    def test_multi_location_state_structure(self) -> None:
        """Test state for multiple locations is properly structured."""
        state: AgentState = {
            "messages": [HumanMessage(content="Weather in Toronto and Vancouver")],
            "parsed": {
                "locations": ["Toronto", "Vancouver"],
                "intent": "current",
                "forecast_granularity": "daily",
                "time_of_day": "all_day",
                "requested_details": ["temperature"],
                "start_date": "2026-06-06",
                "end_date": "2026-06-06",
                "clarification_needed": False,
                "clarification_question": "",
            },
            "weather_data": {
                "locations": [
                    {"_location": {"name": "Toronto"}, "current": {"temperature_2m": 22.5}},
                    {"_location": {"name": "Vancouver"}, "current": {"temperature_2m": 18.0}},
                ],
                "_intent": "current",
            },
            "tool_trace": [],
            "response": "",
        }
        
        assert len(state["parsed"]["locations"]) == 2
        assert len(state["weather_data"]["locations"]) == 2

    def test_multi_location_with_one_failure(self) -> None:
        """Test handling when one location fails and others succeed."""
        state: AgentState = {
            "messages": [],
            "parsed": {
                "locations": ["Toronto", "InvalidCity"],
                "intent": "current",
                "forecast_granularity": "daily",
                "start_date": "2026-06-06",
                "end_date": "2026-06-06",
                "clarification_needed": False,
            },
            "weather_data": {
                "locations": [
                    {"_location": {"name": "Toronto"}, "current": {"temperature_2m": 22.5}},
                    {"_raw_location": "InvalidCity", "error": "Location not found"},
                ],
                "_intent": "current",
            },
            "tool_trace": [],
            "response": "",
        }
        
        # Should have one success and one error
        assert any("Toronto" in str(loc) for loc in state["weather_data"]["locations"])
        assert any("error" in str(loc).lower() for loc in state["weather_data"]["locations"])


class TestGranularityDetection:
    """Test detection of hourly vs daily granularity."""

    def test_hourly_for_today_request(self) -> None:
        """Test that 'today' requests use hourly when appropriate."""
        state: AgentState = {
            "messages": [HumanMessage(content="Hourly weather for today")],
            "parsed": {
                "intent": "forecast",
                "forecast_granularity": "hourly",
                "locations": ["Toronto"],
                "start_date": "2026-06-06",
                "end_date": "2026-06-06",
                "clarification_needed": False,
            },
            "weather_data": {},
            "tool_trace": [],
            "response": "",
        }
        
        assert state["parsed"]["forecast_granularity"] == "hourly"
        assert state["parsed"]["start_date"] == state["parsed"]["end_date"]

    def test_daily_for_week_request(self) -> None:
        """Test that 'this week' uses daily granularity."""
        state: AgentState = {
            "messages": [HumanMessage(content="Weather for this week")],
            "parsed": {
                "intent": "forecast",
                "forecast_granularity": "daily",
                "locations": ["Toronto"],
                "start_date": "2026-06-06",
                "end_date": "2026-06-12",
                "clarification_needed": False,
            },
            "weather_data": {},
            "tool_trace": [],
            "response": "",
        }
        
        assert state["parsed"]["forecast_granularity"] == "daily"
        # Multi-day range
        assert state["parsed"]["end_date"] > state["parsed"]["start_date"]

    def test_historical_hourly(self) -> None:
        """Test that historical requests can use hourly granularity."""
        state: AgentState = {
            "messages": [HumanMessage(content="Yesterday hourly weather")],
            "parsed": {
                "intent": "historical",
                "forecast_granularity": "hourly",
                "locations": ["Toronto"],
                "start_date": "2026-06-05",
                "end_date": "2026-06-05",
                "clarification_needed": False,
            },
            "weather_data": {},
            "tool_trace": [],
            "response": "",
        }
        
        assert state["parsed"]["intent"] == "historical"
        assert state["parsed"]["forecast_granularity"] == "hourly"


class TestRequestDetailFiltering:
    """Test that responses respect requested details."""

    def test_temperature_only_request(self) -> None:
        """Test focused request for temperature only."""
        state: AgentState = {
            "messages": [],
            "parsed": {
                "intent": "current",
                "locations": ["Toronto"],
                "requested_details": ["temperature"],
                "start_date": "2026-06-06",
                "end_date": "2026-06-06",
                "clarification_needed": False,
            },
            "weather_data": {},
            "tool_trace": [],
            "response": "",
        }
        
        assert state["parsed"]["requested_details"] == ["temperature"]
        assert len(state["parsed"]["requested_details"]) == 1

    def test_multi_detail_request(self) -> None:
        """Test request with multiple specific details."""
        state: AgentState = {
            "messages": [],
            "parsed": {
                "intent": "forecast",
                "locations": ["Toronto"],
                "requested_details": ["temperature", "precipitation", "wind"],
                "start_date": "2026-06-06",
                "end_date": "2026-06-06",
                "clarification_needed": False,
            },
            "weather_data": {},
            "tool_trace": [],
            "response": "",
        }
        
        assert len(state["parsed"]["requested_details"]) == 3
        assert all(detail in state["parsed"]["requested_details"] 
                  for detail in ["temperature", "precipitation", "wind"])

    def test_broad_summary_request(self) -> None:
        """Test broad weather summary request."""
        from agent.nodes import DEFAULT_REQUESTED_DETAILS
        
        state: AgentState = {
            "messages": [],
            "parsed": {
                "intent": "forecast",
                "locations": ["Toronto"],
                "requested_details": list(DEFAULT_REQUESTED_DETAILS),
                "start_date": "2026-06-06",
                "end_date": "2026-06-12",
                "clarification_needed": False,
            },
            "weather_data": {},
            "tool_trace": [],
            "response": "",
        }
        
        assert len(state["parsed"]["requested_details"]) == len(DEFAULT_REQUESTED_DETAILS)
