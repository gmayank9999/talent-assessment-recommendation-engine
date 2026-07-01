"""
Centralized configuration for all tunable parameters.

Retrieval weights, LLM settings, and conversation limits live here
so they can be adjusted without touching business logic.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq LLM ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 1500

# --- Retrieval ---
RETRIEVAL_TOP_K: int = 20
SEMANTIC_CANDIDATES: int = 50

# --- Retrieval cache ---
CACHE_ENABLED: bool = True
CACHE_MAX_SIZE: int = 128

# --- Weighted ranking (must sum to 1.0) ---
WEIGHT_SEMANTIC: float = 0.40
WEIGHT_SKILL_MATCH: float = 0.25
WEIGHT_JOB_LEVEL: float = 0.15
WEIGHT_LANGUAGE: float = 0.10
WEIGHT_TEST_CATEGORY: float = 0.10

# --- Conversation limits ---
MAX_CLARIFICATION_TURNS: int = 2
MAX_RECOMMENDATIONS: int = 10
MIN_RECOMMENDATIONS: int = 1

# --- Embedding model ---
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# --- Paths ---
CATALOG_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "catalog.json"
)
EMBEDDINGS_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "embeddings.npz"
)
