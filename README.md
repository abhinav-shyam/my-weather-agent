# Weather Agent

A conversational weather assistant built with Streamlit, LangGraph, Groq, and Open-Meteo.

The app accepts natural-language weather questions, resolves location and time from conversation context, fetches real weather data, and returns a friendly response in a chat interface.

## Features

- Conversational weather queries in plain English
- Structured intent parsing with a typed Pydantic schema
- LangGraph workflow for parse -> fetch -> respond
- Groq `llama-3.3-70b-versatile` for parsing and responses
- Open-Meteo geocoding and weather APIs
- In-memory conversation context for follow-up questions
- Streamlit chat UI with lightweight response streaming
- Basic robustness protections: timeouts, retries, payload validation, and concurrency guards

## Tech Stack

- Python 3.12
- Streamlit
- LangGraph
- LangChain
- Groq
- Open-Meteo
- httpx
- Pydantic
- pytest

## Project Structure

```text
weather-agent/
|-- app.py
|-- agent/
|   |-- __init__.py
|   |-- graph.py
|   |-- llm.py
|   |-- nodes.py
|   |-- prompts.py
|   |-- schema.py
|   `-- tools.py
|-- utils/
|   |-- __init__.py
|   `-- time_utils.py
|-- .env.example
|-- .gitignore
|-- requirements.txt
```

## How It Works

1. The user asks a weather question in Streamlit.
2. The parse node uses a structured LLM output schema to identify:
   - location
   - intent: `current`, `forecast`, or `historical`
   - start and end dates
   - whether clarification is needed
3. The tool node geocodes the location and calls the correct Open-Meteo endpoint.
4. The response node turns raw API data into a concise, user-friendly answer.

## Supported Query Types

- Current weather
- Forecasts
- Historical weather
- Follow-up questions that rely on earlier context

Examples:

- `What's the weather in Bengaluru today?`
- `Will it rain in Tokyo tomorrow?`
- `How hot was Zagreb last week?`
- `What about Delhi?`

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd weather-agent
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
```

You can copy from .env.example

### 5. Run the app

```bash
streamlit run app.py
```

## Environment Variables

| Variable                      | Required | Description                        |
| ----------------------------- | -------- | ---------------------------------- |
| `GROQ_API_KEY`                | Yes      | API key for Groq                   |
| `LLM_REQUEST_TIMEOUT_SECONDS` | No       | Per-request LLM timeout in seconds |

## Reliability Notes

The current implementation includes:

- structured output for intent parsing
- HTTP timeouts for external API calls
- retry logic with backoff for transient API/LLM failures
- basic response validation for geocoding and weather payloads
- configurable timeout and retry handling for Groq calls
- lightweight concurrency limits to reduce rate-limit pressure

## Current Scope

This repository currently implements Phase A:

- single-session chat memory
- no persistent database
- no multi-thread chat UI
- no Docker setup yet

Phase B is planned for persistent history, threads, and deployment support.

## Development Notes

- Keep `.env`, `.venv/` out of version control.
- Use `.env.example` to document required configuration.
- If your IDE reports unresolved imports, confirm it is using the project virtual environment.
- Open-Meteo does not require an API key.

## Known Limitations

- Chat history resets when the Streamlit session is refreshed
- Streaming is currently simulated word-by-word in the UI
- Historical weather routing is optimized for Open-Meteo's forecast/archive boundary, but still depends on API availability

## Roadmap

- True token streaming with LangGraph `astream_events`
- Persistent chat history
- Sidebar thread management
- Docker support
- Deployment-ready production setup

## License

Add a license for this project if you plan to publish or share it publicly.
