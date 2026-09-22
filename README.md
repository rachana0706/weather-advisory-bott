# Weather-Advisory Support Bot

A LangGraph-based conversational bot that provides outdoor activity safety advice using **live weather data** and explicitly defined **Standard Operating Procedures (SOPs)**.

The system is designed so that the LLM handles natural-language understanding and response composition, while **weather retrieval and safety-policy decisions remain deterministic and traceable**.

---

## 1. Architecture

```text
User Message
     │
     ▼
┌─────────────────────┐
│  Intent Extraction  │
│   Local Llama 2     │
└──────────┬──────────┘
           │
           ▼
     Location found?
       /          \
     No            Yes
     │              │
     ▼              ▼
Ask for        Open-Meteo
Location        Weather API
                    │
              API successful?
                /        \
              No          Yes
              │            │
              ▼            ▼
          API Error     SOP Matcher
                         │
                   Policy matched?
                    /          \
                  No            Yes
                  │              │
                  ▼              ▼
            No Guidance      Generate Advice
                              using facts
```

The application is implemented as a real **LangGraph state graph with conditional branches**, rather than a simple linear chain.

---

## 2. Architecture Highlights

### Local LLM

The application uses **Ollama** with the locally hosted `llama2:latest` model.

The LLM is used for:

* extracting user intent
* identifying location, activity, time reference, and vulnerable groups
* composing the final natural-language response

The LLM does **not** determine which safety policy applies.

### Deterministic Weather Retrieval

Weather is retrieved from the **Open-Meteo API**.

The application:

1. Geocodes the requested location.
2. Retrieves live forecast data using the resulting latitude and longitude.
3. Extracts the weather metrics required by the SOP matcher.
4. Passes those actual API values to the deterministic policy engine.

The application does not invent weather values when the API does not provide them.

### Deterministic SOP Matcher

Safety policies are stored in:

```text
sops.yaml
```

The matcher evaluates:

* activity
* vulnerable group
* weather metrics
* comparison operators
* severity
* priority
* fallback policies

The matcher is implemented using ordinary Python logic and does not ask the LLM to make safety decisions.

### Fact-Based Response Generation

After a policy is selected, the LLM receives the actual weather data and the selected SOP guidance.

The generated response is instructed to:

* use only the supplied weather facts
* explicitly mention the relevant weather values
* provide the selected SOP guidance
* identify the applicable SOP

Therefore, the model is used primarily for **language composition rather than policy reasoning**.

### No-Guidance Handling

If no specific SOP applies, the bot does not invent generic safety advice.

Instead, it reports the available forecast information and explicitly states that no specific safety advisory was triggered.

### Session Memory

LangGraph's `MemorySaver` checkpointer maintains conversation state within a session.

This allows later messages to reuse previously extracted context, such as location or activity.

---

## 3. SOP Design

The SOPs are defined in YAML because this keeps policy configuration separate from application logic.

Each SOP can define:

* `id`
* `category`
* `severity`
* `priority`
* applicable activities
* vulnerable groups
* weather conditions
* fallback behavior
* guidance

Example:

```yaml
- id: SOP-001
  category: outdoor_exercise
  severity: 3
  priority: 80
  conditions:
    activities: [cycling, biking, running]
    weather:
      wind_speed_10m:
        operator: ">"
        value: 40
  guidance: "Advise avoiding these activities during severe winds."
```

### Why YAML?

YAML was chosen because it is:

* human-readable
* easy to edit during policy review
* structured enough to represent conditions and metadata
* independent of the weather retrieval and LLM implementation

This allows new or modified policies to be added without changing the weather code, matcher control flow, or LLM code.

For example, an additional SOP can be added to `sops.yaml` and the generic matcher can evaluate it without adding a new `if sop_id == ...` branch.

### Fallback / Qualitative Policy

The system also supports policies where no numeric weather threshold is required.

For example, `SOP-011` acts as a fallback for picnic and outdoor-gathering questions. It allows the system to provide a forecast-based suitability description when no more specific safety policy applies.

Specific weather safety policies take precedence over the fallback.

---

## 4. Multiple SOP Matches

More than one SOP can apply to the same weather conditions.

The matcher handles this deliberately rather than selecting a rule arbitrarily.

Conflict resolution is based on:

1. **Highest severity**
2. **Highest priority** when severity is equal

The selected SOP and the matched SOP IDs are retained in the graph state so the final response remains traceable to the policy decision.

Fallback policies are only used when no specific policy applies.

---

## 5. Project Structure

```text
weather-advisory-bott/
│
├── src/
│   ├── graph.py              # LangGraph workflow and branching
│   ├── matcher.py            # Deterministic SOP matching
│   ├── nodes.py              # LLM, weather and response nodes
│   ├── state.py              # LangGraph state definition
│   └── weather.py            # Geocoding and Open-Meteo integration
│
├── tests/
│   ├── eval_suite.py         # Assignment evaluation suite
│   └── test_matcher.py       # Matcher unit tests
│
├── app.py                    # Streamlit frontend
├── sops.yaml                 # Safety policy configuration
├── requirements.txt          # Python dependencies
├── README.md
└── .gitignore
```

---

# 6. Setup

## Backend Setup

Create and activate a Python virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Ollama Setup

Install Ollama separately and make sure it is available on the system.

Download the required model:

```bash
ollama pull llama2:latest
```

Make sure the Ollama service is running.

On systems where the Ollama service is not already running:

```bash
ollama serve
```

The application expects the model:

```text
llama2:latest
```

The model runs locally; no Gemini API key is required.

---

# 8. Frontend / Chat Interface

The project uses **Streamlit** as its minimal chat frontend.

Start the application with:

```bash
streamlit run app.py
```

The Streamlit interface sends user messages to the LangGraph application and displays the resulting advisory.

The conversation maintains LangGraph session state using the checkpointer.

---

# 9. Evaluation Suite

The evaluation suite is located at:

```text
tests/eval_suite.py
```

It covers the major requirements of the assignment, including:

| Evaluation                  | Purpose                                                        |
| --------------------------- | -------------------------------------------------------------- |
| Clear SOP application       | Verifies that applicable policies are selected                 |
| Paraphrased intents         | Checks natural-language variation                              |
| Live weather grounding      | Uses real Open-Meteo data                                      |
| Severe-condition resolution | Checks multiple matching SOPs and priority/severity resolution |
| No-SOP case                 | Verifies honest no-guidance behavior                           |
| Unreachable API             | Verifies graceful weather API failure handling                 |
| Adversarial input           | Checks resistance to instruction/prompt injection              |
| Tomorrow reference          | Checks temporal intent extraction                              |
| Session memory              | Checks retention of conversational context                     |

The matcher also has a separate unit-test suite in:

```text
tests/test_matcher.py
```

---

## 10. Running Tests

Run the matcher tests:

```bash
python -m pytest tests/test_matcher.py -v
```

Run the full evaluation suite:

```bash
python -m pytest tests/eval_suite.py -v -s
```

### Current Result

The evaluation suite currently passes:

```text
11 passed
```

The matcher test suite also passes completely.

The evaluation tests mock LLM calls where deterministic behavior is required, so the full automated evaluation suite does not depend on Llama 2 generating identical wording on every run.

---

## 11. Live Weather Evaluation Note

The live-weather evaluation uses the real Open-Meteo service rather than hard-coded weather events.

Because weather forecasts change over time, a live-weather test is inherently time-dependent. A condition that is severe during one run may no longer be present later.

The evaluation therefore verifies that the application correctly grounds its behavior in the weather values returned by the API rather than assuming a permanently fixed forecast.

Synthetic weather data is used separately where deterministic testing of specific policy combinations and conflict resolution is required.

---

## 12. Safety and Traceability Principles

The application follows these principles:

### Weather facts come from the API

The LLM is not allowed to invent forecast values.

### Safety decisions come from SOPs

The LLM does not create new safety thresholds or policies.

### Every advisory is traceable

A response either:

* identifies the applicable SOP, or
* explicitly states that no specific SOP applies.

### API failures are handled explicitly

If weather retrieval fails, the application returns an API-failure response instead of fabricating forecast information.

### Policy changes remain separate from application logic

SOPs can be added or modified in `sops.yaml` without modifying the weather retrieval or LLM implementation.

---

## 13. Dependencies

Main dependencies include:

* **LangGraph** - stateful graph orchestration
* **LangChain Core** — LLM message abstractions
* **LangChain Ollama** — Ollama integration
* **Ollama / Llama 2** — local language model
* **Requests** — Open-Meteo API requests
* **PyYAML** — SOP configuration
* **Pydantic** — structured intent validation
* **Streamlit** — chat frontend
* **Pytest** — evaluation and unit testing

See `requirements.txt` for pinned Python package versions.

---

## 14. Main Design Goal

The central design principle is:

> **The model talks, the weather API provides facts, and the SOP engine makes the policy decision.**

This separation makes the bot more predictable, testable, and auditable than an architecture where the LLM directly decides what weather conditions are safe.
