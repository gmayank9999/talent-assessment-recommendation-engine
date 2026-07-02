# Talent Assessment Recommendation Engine

A conversational AI agent that recommends technical and behavioral assessments through multi-turn dialogue. Built as a stateless FastAPI service backed by hybrid retrieval (semantic search + keyword matching + business rules) and Groq-hosted LLM reasoning.

**Live Demo:** [https://talent-assessment-recommendation-engine.onrender.com](https://talent-assessment-recommendation-engine.onrender.com) (Expect a ~40s cold start if the free tier is asleep).

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
      - Semantic vector search (all-MiniLM-L6-v2)
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

## Setup & Deployment

### Run Locally

1. **Prerequisites**: Python 3.11+, Groq API key (free tier works)
2. **Install**:
   ```bash
   git clone https://github.com/gmayank9999/talent-assessment-recommendation-engine.git
   cd talent-assessment-recommendation-engine
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Set API Key**: Create a `.env` file and add `GROQ_API_KEY=your_key_here`
4. **Build Vector Index**: `python scripts/build_index.py` (Runs once to index catalog)
5. **Start Server**: `uvicorn app.main:app --reload --port 8000`

### Deploy on Render (Free Tier)

This repository includes a `render.yaml` blueprint. To deploy:
1. Connect this GitHub repo in Render.
2. The blueprint will auto-configure a Python 3.11 environment.
3. Add `GROQ_API_KEY` in the Render Environment Variables tab.
4. Deploy! The build script (`build.sh`) uses CPU-only PyTorch to stay within free-tier limits.

## API Usage

**GET /health**
```bash
curl https://talent-assessment-recommendation-engine.onrender.com/health
# {"status": "ok"}
```

**POST /chat**
```bash
curl -X POST https://talent-assessment-recommendation-engine.onrender.com/chat \
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
  "reply": "For a senior Java developer, I recommend...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Key Design Decisions

1. **Stateless architecture**: Every `/chat` call receives the full conversation history. No server-side session state is maintained.
2. **Conversation state machine**: Six deterministic states (CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE, CONFIRM) with rule-based classification. This prevents unpredictable LLM loops.
3. **Hybrid retrieval**: Combines semantic search, keyword matching, and business-rule filtering. Pure semantic search often confuses structurally similar products (e.g., Core Java Entry vs Advanced) and dilutes attention on multi-technology JDs.
4. **Post-processor + validator separation**: Normalization (URL formatting, canonical names) is handled separately from correctness checks (strict catalog membership). Makes both independently testable.
5. **Anti-Hallucination Gate**: The validator intercepts the LLM's response right before returning to the user and strips any URL or assessment name that does not strictly exist in the offline catalog dataset.

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
scripts/
  build_index.py         # one-time index builder
tests/
  demo.py                # Interactive CLI for testing all chat states
  test_conversation_parser.py  # unit tests
  test_trace_replay.py         # integration test harness
```
