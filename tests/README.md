# Weather Agent Test Suite

Comprehensive pytest suite validating critical system behavior of the weather agent.

## Test Coverage

### 107 Total Tests

- **test_formatting.py** (51 tests) — Data formatting and helper functions
- **test_integration.py** (18 tests) — Integration workflows and state management
- **test_nodes.py** (12 tests) — Node functions and message handling
- **test_tools.py** (21 tests) — Tool validation and error handling

---

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_formatting.py
```

### Run specific test class
```bash
pytest tests/test_formatting.py::TestNormalizeLocations
```

### Run specific test
```bash
pytest tests/test_formatting.py::TestNormalizeLocations::test_single_location
```

### Run tests matching a pattern
```bash
pytest tests/ -k "parse"
```

### Run with coverage report
```bash
pytest tests/ --cov=agent --cov-report=html
```

---

## Test Sections

### 1. Formatting Tests (`test_formatting.py`)

Validates data transformation and presentation logic:

- **Location normalization** — Deduplication, whitespace handling, type validation
- **Greeting detection** — Case-insensitive greeting recognition
- **Weather code descriptions** — WMO code interpretation to human-readable text
- **Wind phrase generation** — Speed-based wind descriptions
- **Sample indexing** — Downsampling large datasets for display
- **Hourly/Daily/Current formatting** — Response formatting for different time granularities
- **Multi-location responses** — Handling multiple locations in a single request
- **Location label extraction** — Fallback logic for location names
- **Error message formatting** — User-friendly error presentation

### 2. Node Tests (`test_nodes.py`)

Tests core agent workflow nodes:

- **Intent parsing** (`parse_intent`)
  - Successful parsing with mocked LLM
  - Error handling and fallback messages
  - Non-weather request detection
  - Automatic clarification when locations missing
  
- **Non-weather handling** (`handle_non_weather`)
  - Greeting detection and response
  - Clarification for non-weather requests
  - Tool trace updates

- **State management**
  - Immutable state updates
  - Message accumulation
  - Tool trace accumulation

- **Message handling**
  - Last user message extraction
  - Empty message list handling

### 3. Integration Tests (`test_integration.py`)

End-to-end workflow validation:

- **Graph construction** — Verify graph builds successfully
- **State flow** — Proper state structure and field populations
- **Error handling** — Graceful error handling and state resilience
- **Multi-location workflows** — Multiple location request handling
- **Date range handling** — Single day, forecast, and historical requests
- **Tool tracing** — Debugging trace structure and error capture
- **Requested detail variations** — Different weather data requests (temperature-only, full details, etc.)

### 4. Tool Tests (`test_tools.py`)

Validates tool functions and API response handling:

- **Retry logic** — Identification of retryable vs permanent errors
  - Network timeouts and errors (retryable)
  - Rate limits (retryable)
  - Server errors (retryable)
  - Invalid requests (not retryable)

- **Geocoding validation** (`_validate_geocode_payload`)
  - Valid responses with all required fields
  - Missing latitude/longitude handling
  - Empty results handling
  - Default timezone assignment

- **Weather validation** (`_validate_weather_payload`)
  - Current weather validation
  - Daily forecast validation (with time series)
  - Hourly forecast validation
  - Missing data detection
  - Error payload handling

---

## Test Fixtures

Available in `conftest.py`:

```python
# Minimal valid state
basic_state

# State with parsed weather intent
parsed_state

# Mock current weather data
weather_data_fixture

# Mock daily forecast data
forecast_data_fixture

# Mock hourly forecast data
hourly_data_fixture

# State with non-weather intent
non_weather_state

# State with greeting message
greeting_state
```

---

## Configuration

Test configuration in `pytest.ini`:

- **asyncio_mode**: Auto mode for async test support
- **testpaths**: Tests directory
- **python_files**: `test_*.py` pattern
- **Markers**: Categories like `@pytest.mark.asyncio`
- **Warnings**: Filtered to reduce noise

---

## Key Testing Patterns

### State Testing
```python
def test_state_immutability(basic_state):
    original_response = basic_state.get("response", "")
    updated_state = {**basic_state, "response": "New"}
    assert basic_state["response"] == original_response
```

### Async Testing
```python
@pytest.mark.asyncio
async def test_parse_intent(basic_state, monkeypatch):
    monkeypatch.setattr("agent.nodes.call_structured_llm", mock_fn)
    result = await parse_intent(basic_state)
    assert result["parsed"]["intent"] == "current"
```

### Fixture Usage
```python
def test_format_response(weather_data_fixture):
    response = _format_current_response(weather_data_fixture)
    assert "Toronto" in response
```

---

## Critical Behaviors Validated

✅ **Data Integrity**
- Location deduplication and normalization
- State immutability across updates
- Proper message accumulation

✅ **Error Handling**
- Graceful LLM failures with fallback messages
- Retryable vs permanent error distinction
- Missing data handling in API responses

✅ **Intent Parsing**
- Weather request classification
- Clarification detection for missing locations
- Non-weather request routing

✅ **Multi-Location Support**
- Concurrent location fetching
- Error isolation per location
- Proper response aggregation

✅ **Data Formatting**
- Appropriate detail selection
- Human-readable weather descriptions
- Timezone-aware hour formatting

✅ **Tool Validation**
- Geocoding response validation
- Weather data completeness checks
- API error detection

---

## Running in CI/CD

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest-asyncio pytest-cov

# Run tests with coverage
pytest tests/ --cov=agent --cov-report=term-missing

# Run only fast unit tests
pytest tests/ -m "not slow"

# Generate HTML coverage report
pytest tests/ --cov=agent --cov-report=html
```

---

## Extending the Tests

To add new tests:

1. Create test file following `test_*.py` pattern
2. Use fixtures from `conftest.py` or create new ones
3. Mark async tests with `@pytest.mark.asyncio`
4. Follow naming convention: `Test*` classes, `test_*` methods
5. Run `pytest tests/` to verify

---

## Dependencies

- **pytest** >= 9.0.0 — Test framework
- **pytest-asyncio** — Async test support
- **langchain** — For agent functionality mocking
- **python** >= 3.12.4
