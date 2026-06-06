from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Defines the state structure for the weather agent.
    This state is passed through the graph nodes and can be updated at each step."""

    messages: Annotated[list[BaseMessage], add_messages]
    parsed: dict[str, Any]
    weather_data: dict[str, Any]
    tool_trace: list[Any]
    response: str
