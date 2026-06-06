from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.nodes import (
    AgentState,
    fetch_weather_node,
    generate_response,
    handle_non_weather,
    parse_intent,
)


def _route_after_parse(state: AgentState) -> str:
    parsed = state.get("parsed", {})

    intent = parsed.get("intent")

    if intent == "non_weather":
        return "handle_non_weather"
    if parsed.get("clarification_needed"):
        return END
    if intent in {"current", "forecast", "historical"}:
        return "fetch_weather"
    return END


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_intent", parse_intent)
    graph.add_node("fetch_weather", fetch_weather_node)
    graph.add_node("handle_non_weather", handle_non_weather)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("parse_intent")

    graph.add_conditional_edges(
        "parse_intent",
        _route_after_parse,
    )
    graph.add_edge("fetch_weather", "generate_response")
    graph.add_edge("handle_non_weather", END)
    graph.add_edge("generate_response", END)

    return graph.compile()


weather_agent = build_graph()
