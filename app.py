from __future__ import annotations

import asyncio
import time
from typing import Generator

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import weather_agent
from agent.nodes import AgentState


def stream_response(text: str) -> Generator[str, None, None]:
    """Yield a response word-by-word for a lightweight streaming effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def run_agent(initial_state: AgentState) -> AgentState:
    return asyncio.run(weather_agent.ainvoke(initial_state))

def main() -> None:
    st.set_page_config(page_title="Weather Agent", page_icon="🌤", layout="centered")
    st.title("🌤 Weather Agent")
    st.caption("Ask me about weather anywhere in the world.")

    ensure_session_state()
    render_chat_history()

    if prompt := st.chat_input("Ask about the weather..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.agent_messages.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        initial_state: AgentState = {
            "messages": list(st.session_state.agent_messages),
            "parsed": {},
            "weather_data": {},
            "tool_trace": [],
            "response": "",
        }

        try:
            with st.spinner("Checking the weather..."):
                final_state = run_agent(initial_state)
            assistant_response = final_state.get(
                "response",
                "I wasn't able to fetch the weather right now. Please try again shortly.",
            )
            st.session_state.agent_messages = final_state.get(
                "messages",
                list(st.session_state.agent_messages)
                + [AIMessage(content=assistant_response)],
            )
        except Exception:
            assistant_response = (
                "I ran into a temporary issue while checking the weather. Please try again shortly."
            )
            st.session_state.agent_messages.append(
                AIMessage(content=assistant_response)
            )

        with st.chat_message("assistant"):
            st.write_stream(stream_response(assistant_response))

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )


if __name__ == "__main__":
    main()
