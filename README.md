# SHL Assessment Recommendation Engine

A conversational agent that recommends SHL Individual Test Solutions through multi-turn dialogue. Built as a stateless FastAPI service backed by hybrid retrieval (semantic search + keyword matching + business rules) and Groq-hosted LLM reasoning.

## Architecture

```
User -> FastAPI /chat
          |
    Conversation Parser (deterministic, extracts shortlist + state)
          |
    Requirement Extractor (one LLM call for structured extraction)
          |
    Intent Classifier (rule-based: CLARIFY/RECOMMEND/REFINE/COMPARE/REFUSE/CONFIRM)
          |
    Hybrid Retrieval Engine
      - Semantic vector search (ChromaDB + all-MiniLM-L6-v2)
      - Keyword/exact match boost
      - Business rule hard filters
      - Weighted candidate ranking
          |
    Reasoning Engine (one LLM call via Groq, modular prompt)
          |
    Post-Processor (normalize URLs, names, test_types against catalog)
          |
    Validator (hard gate: strip any hallucinated entries)
          |
    JSON Response
```

## Setup

### Prerequisites

- Python 3.11+
- A Groq API key (free tier works)

### Installation

```bash
# clone the repo
git clone https://github.com/gmayank9999/talent-assessment-recommendation-engine.git
cd talent-assessment-recommendation-engine

# create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt

# set up environment variables
cp .env.example .env
# edit .env and add your GROQ_API_KEY
```

### Build the vector index

```bash
python scripts/build_index.py
```

This builds the ChromaDB index from the product catalog. Only needs to run once (or when the catalog changes).

### Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### API Endpoints

**GET /health**
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

**POST /chat**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need assessments for a senior Java developer"}
    ]
  }'
```

Response:
```json
{
  "reply": "...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

### Run tests

```bash
# unit tests (no server required)
pytest tests/test_conversation_parser.py -v

# trace replay (requires running server)
pytest tests/test_trace_replay.py -v
```

### Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key shl-recommender
```

## Key Design Decisions

1. **Stateless architecture**: every /chat call receives the full conversation history. No server-side session state.

2. **Conversation state machine**: six deterministic states (CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE, CONFIRM) with rule-based classification. No second LLM call for intent detection.

3. **Hybrid retrieval**: combines semantic search, keyword matching, and business-rule filtering. Pure semantic search confuses similar products (e.g., Core Java Entry vs Advanced) and dilutes attention on multi-technology JDs.

4. **OPQ32r default inclusion**: personality assessment is included by default for professional/technical roles (matching trace behavior across C1, C2, C4, C5, C7, C8, C9) with an explicit opt-out offer.

5. **Post-processor + validator separation**: normalization (URL formatting, canonical names) is handled separately from correctness checks (catalog membership). Makes both independently testable.

6. **`recommendations: null` vs `[]`**: uses null (matching the 10 reference traces) rather than empty array (mentioned in the spec). Documented as a deliberate design decision.

## Project Structure

```
app/
  main.py              # FastAPI endpoints and orchestration
  config.py            # centralized configuration
  schemas.py           # Pydantic request/response models
  catalog.py           # catalog loader and lookup structures
  conversation_parser.py  # deterministic conversation state extraction
  requirement_extractor.py # structured requirement extraction (LLM)
  intent_classifier.py    # rule-based intent classification
  retrieval.py           # hybrid retrieval pipeline
  reasoning.py           # modular prompt assembly and LLM call
  post_processor.py      # recommendation normalization
  validator.py           # hallucination prevention gate
data/
  chroma_index/          # persisted vector index (built from catalog)
scripts/
  build_index.py         # one-time index builder
tests/
  test_conversation_parser.py  # unit tests
  test_trace_replay.py         # trace replay against live service
```
