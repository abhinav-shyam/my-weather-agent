"""Tests for agent node functions."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import (
    handle_non_weather,
    parse_intent,
)
from agent.schema import WeatherIntent
from agent.state import AgentState


class TestParseIntent:
    """Test intent parsing node."""

    @pytest.mark.asyncio
    async def test_parse_with_mocked_llm(self, basic_state, monkeypatch) -> None:
        """Test parse_intent with mocked LLM."""

        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=["Toronto"],
                intent="current",
                start_date="2026-06-06",
                end_date="2026-06-06",
                clarification_needed=False,
                clarification_question="",
            )

        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)

        result = await parse_intent(basic_state)

        assert result["parsed"]["intent"] == "current"
        assert "Toronto" in result["parsed"]["locations"]
        assert result["parsed"]["clarification_needed"] is False
        assert "tool_trace" in result
        assert len(result["tool_trace"]) > 0

    @pytest.mark.asyncio
    async def test_parse_with_llm_error(self, basic_state, monkeypatch) -> None:
        """Test parse_intent handles LLM errors gracefully."""

        async def mock_call_structured_llm(messages):
            raise RuntimeError("LLM unavailable")

        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)

        result = await parse_intent(basic_state)

        assert result["parsed"]["clarification_needed"] is True
        assert len(result["messages"]) > 0
        assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_parse_non_weather_intent(self, greeting_state, monkeypatch) -> None:
        """Test parse_intent with non-weather request."""

        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=[],
                intent="non_weather",
                start_date="2026-06-06",
                end_date="2026-06-06",
                clarification_needed=False,
                clarification_question="",
            )

        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)

        result = await parse_intent(greeting_state)

        assert result["parsed"]["intent"] == "non_weather"

    @pytest.mark.asyncio
    async def test_parse_requires_clarification(self, basic_state, monkeypatch) -> None:
        """Test parse_intent detects when clarification is needed."""

        async def mock_call_structured_llm(messages):
            return WeatherIntent(
                locations=[],  # No locations parsed
                intent="forecast",
                start_date="2026-06-06",
                end_date="2026-06-06",
                clarification_needed=False,
                clarification_question="",
            )

        monkeypatch.setattr("agent.nodes.call_structured_llm", mock_call_structured_llm)

        result = await parse_intent(basic_state)

        assert result["parsed"]["clarification_needed"] is True
        assert "Which location" in result["messages"][0].content


class TestHandleNonWeather:
    """Test non-weather request handling."""

    @pytest.mark.asyncio
    async def test_greeting_response(self, greeting_state) -> None:
        """Test that greetings are handled appropriately."""
        result = await handle_non_weather(greeting_state)

        assert isinstance(result["messages"][0], AIMessage)
        content = result["messages"][0].content
        assert "weather assistant" in content.lower()

    @pytest.mark.asyncio
    async def test_non_weather_response(self, non_weather_state) -> None:
        """Test non-weather request gets redirected."""
        result = await handle_non_weather(non_weather_state)

        assert isinstance(result["messages"][0], AIMessage)
        assert result["response"]
        assert len(result["tool_trace"]) > 0

    @pytest.mark.asyncio
    async def test_tool_trace_updated(self, non_weather_state) -> None:
        """Test that tool trace is properly updated."""
        initial_trace_len = len(non_weather_state.get("tool_trace", []))
        result = await handle_non_weather(non_weather_state)

        assert len(result["tool_trace"]) > initial_trace_len
        assert {"route": "non_weather"} in result["tool_trace"]


class TestStateManagement:
    """Test agent state handling."""

    def test_state_immutability(self, basic_state) -> None:
        """Test that state updates don't mutate original."""
        original_response = basic_state.get("response", "")

        updated_state = {
            **basic_state,
            "response": "Updated response",
        }

        assert basic_state["response"] == original_response
        assert updated_state["response"] == "Updated response"

    def test_messages_accumulation(self) -> None:
        """Test that messages accumulate in state."""
        state: AgentState = {
            "messages": [HumanMessage(content="First")],
            "parsed": {},
            "weather_data": {},
            "tool_trace": [],
            "response": "",
        }

        state_with_reply = {
            **state,
            "messages": [*state["messages"], AIMessage(content="Reply")],
        }

        assert len(state_with_reply["messages"]) == 2
        assert isinstance(state_with_reply["messages"][1], AIMessage)

    def test_tool_trace_accumulation(self) -> None:
        """Test that tool trace entries accumulate."""
        state: AgentState = {
            "messages": [],
            "parsed": {},
            "weather_data": {},
            "tool_trace": [{"step": 1}],
            "response": "",
        }

        state_with_trace = {
            **state,
            "tool_trace": [*state["tool_trace"], {"step": 2}],
        }

        assert len(state_with_trace["tool_trace"]) == 2
        assert state_with_trace["tool_trace"][-1]["step"] == 2


class TestMessageHandling:
    """Test message handling in nodes."""

    def test_last_user_message_extraction(self, basic_state) -> None:
        """Test extracting the last user message."""
        from agent.nodes import _last_user_message

        messages = [
            HumanMessage(content="First question"),
            AIMessage(content="First response"),
            HumanMessage(content="Second question"),
        ]

        state_with_messages: AgentState = {
            **basic_state,
            "messages": messages,
        }

        last_message = _last_user_message(state_with_messages["messages"])
        assert last_message == "Second question"

    def test_empty_messages(self, basic_state) -> None:
        """Test handling empty message list."""
        from agent.nodes import _last_user_message

        last_message = _last_user_message([])
        assert last_message == ""
