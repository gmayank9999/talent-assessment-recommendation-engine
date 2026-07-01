# Approach Document: SHL Assessment Recommendation Agent

## 1. Design Choices and Rationale

### Conversation State Machine

The agent uses an explicit six-state conversation model (CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE, CONFIRM) with rule-based classification rather than allowing the LLM to decide what to do freely. This gives deterministic control over when the agent asks questions vs. recommends, when it refuses off-topic requests, and when it terminates the conversation. The intent classifier runs before the LLM call, so the LLM receives specific behavioral instructions for the classified state rather than making that decision itself.

### Conversation Parser as a Dedicated Pre-Step

Before any requirement extraction or intent classification, a deterministic parser scans the raw message history to extract the current shortlist, count clarification turns, and identify explicit add/remove instructions. This is essential for the REFINE behavior: when a user says "Add AWS and Docker. Drop REST", the system needs to know exactly which items were in the previous shortlist so it can perform a delta operation rather than regenerating from scratch. Relying on the LLM to re-derive the shortlist from history would be fragile and inconsistent.

### Hybrid Retrieval Over Pure Semantic Search

Pure embedding-based search fails in two specific scenarios observed in the reference traces:

1. **Similar products with critical distinctions**: "Core Java (Entry Level)" and "Core Java (Advanced Level)" have nearly identical embeddings but serve different seniority levels. Job-level metadata filtering resolves this.

2. **Multi-technology JDs**: A job description mentioning seven technologies diffuses the embedding signal across all of them. Keyword matching on exact technology names reliably surfaces all relevant tests.

The retrieval pipeline combines semantic search (ChromaDB with all-MiniLM-L6-v2), keyword boosting, job-level/language/exclusion hard filters, and a weighted scoring function with configurable weights.

### OPQ32r Default Inclusion

Across traces C1, C2, C4, C5, C7, C8, and C9, the OPQ32r personality assessment is included by default for professional/managerial/technical roles, with the agent explicitly flagging it as a default and offering to remove it. This is implemented as an explicit retrieval-layer rule rather than emergent LLM behavior. It directly improves recall on the reference traces and demonstrates the expected "flag-and-offer-to-remove" behavior pattern.

### Post-Processor and Validator Separation

Normalization (URL trailing-slash consistency, canonical name mapping, test_type derivation) is handled in a post-processor layer, separate from the validator which does binary pass/fail catalog membership checks. An entry with the right URL but a missing trailing slash should be fixed, not rejected. Keeping these concerns separate makes both independently testable.

## 2. Retrieval Setup

The retrieval pipeline has five stages:

1. **Cache check**: LRU cache keyed on a hash of the structured requirements (role, skills, job level, languages, excluded tests). Avoids redundant retrieval for similar queries.

2. **Semantic vector search**: all-MiniLM-L6-v2 (22M params, runs locally, no external API call at query time) embedded in ChromaDB with cosine similarity. Top-50 candidates retrieved.

3. **Keyword boost**: exact technology name matching against catalog names and descriptions. Merged with semantic results.

4. **Business-rule hard filters**: job-level compatibility, language support, explicit exclusions.

5. **Weighted ranking**: configurable weights (semantic: 0.40, skill match: 0.25, job level: 0.15, language: 0.10, test category: 0.10). Weights are centralized in config.py and tunable without code changes.

The embedding model and ChromaDB index are loaded once at startup. Per-request retrieval latency is sub-500ms.

## 3. Prompt Design

The LLM call uses a modular prompt assembled from five blocks:

- **System role**: scope constraints, anti-hallucination instructions, prompt-injection defense
- **Conversation state**: classified intent, extracted requirements, current shortlist
- **Candidate pool**: top-20 retrieval results with full metadata
- **Behavior instructions**: loaded dynamically for the classified intent only
- **Output format**: strict JSON schema with field-level rules

Only one LLM call per turn. JSON mode is enforced via Groq's `response_format` parameter.

## 4. The `recommendations: null` vs `[]` Decision

The assignment spec's example shows `recommendations: []` for turns without recommendations. However, all 10 reference traces use `null`. Since the traces are the more concrete and numerous source of truth, the implementation uses `Optional[List[Recommendation]]` with null on CLARIFY, COMPARE, and REFUSE turns. This is a deliberate design choice.

## 5. Evaluation Approach and Results

### Methodology

A trace replay harness replays each of the 10 sample conversations (C1-C10) against the running service. Each turn is sent with the full accumulated history, and the output is checked for:
- Correct recommendation timing (null vs. populated at each turn)
- `end_of_conversation` flag timing
- Recall against the reference shortlists
- Shortlist continuity on REFINE turns

### Results

- **Hallucination rate**: 0.0 post-validation (structurally guaranteed by the validator)
- **Schema compliance**: 1.0 (every response passes Pydantic validation)
- **Recommendation timing**: matches reference traces across all 10 conversations
- **Recall@10**: varies by trace complexity; strongest on technology-specific traces (C8, C9), adequate on broad leadership traces (C1, C5)

## 6. What Did Not Work and How It Was Fixed

- **Pure semantic search confused Core Java variants**: the Entry-Level and Advanced-Level tests have similar embeddings. Adding keyword matching and job-level filtering resolved this by matching on exact product names and filtering by the candidate's stated seniority.

- **Infinite clarification loops**: early versions would keep asking clarifying questions without converging. Adding a `clarification_turns_used` counter with a forced RECOMMEND after 2 turns fixed this, matching the reference trace pattern (C9 asks exactly 2 questions before recommending).

- **LLM occasionally hallucinated product names**: the post-processor normalizes against canonical catalog entries, and the validator strips anything that does not pass URL/name membership. Together they guarantee a 0.0 hallucination rate in the final output.

- **REFINE regenerating the full shortlist**: without the conversation parser extracting the exact previous shortlist, the LLM would sometimes regenerate all items from scratch on a REFINE turn. Passing the parsed shortlist directly into the prompt and instructing delta-only behavior fixed this.

## 7. AI Tools Used

AI assistance was used for code scaffolding and debugging individual components. The following were independently designed: the six-state conversation state machine architecture, the hybrid retrieval scoring function and weight selection, the modular prompt structure with intent-specific behavior blocks, the conversation parser as a dedicated pre-step, and the post-processor/validator separation. All design decisions are grounded in analysis of the 10 reference conversation traces.
