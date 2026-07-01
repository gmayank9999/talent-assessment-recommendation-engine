"""
Hybrid retrieval pipeline.

Combines semantic vector search, keyword matching, business-rule hard filters,
and weighted candidate ranking to produce a grounded candidate pool for the LLM.

The pipeline runs in this order:
    0. Cache check (skip everything if we have a cached result)
    1. Build query signals from structured requirements
    2. Semantic search via pre-computed embeddings + cosine similarity
    3. Keyword/exact-match boost for technology names
    4. Business-rule hard filters (job level, language, exclusions)
    5. Weighted candidate ranking
    6. OPQ32r default-inclusion heuristic

Uses numpy dot product on pre-normalized embeddings for semantic search.
With ~600 catalog entries, this is faster and simpler than a full vector DB.
"""

import hashlib
import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    RETRIEVAL_TOP_K,
    SEMANTIC_CANDIDATES,
    CACHE_ENABLED,
    CACHE_MAX_SIZE,
    WEIGHT_SEMANTIC,
    WEIGHT_SKILL_MATCH,
    WEIGHT_JOB_LEVEL,
    WEIGHT_LANGUAGE,
    WEIGHT_TEST_CATEGORY,
)
from app.catalog import CatalogStore, CatalogEntry

logger = logging.getLogger(__name__)

# adjacent job levels for partial scoring (0.5 instead of 0.0)
ADJACENT_LEVELS = {
    "Entry-Level": {"Graduate"},
    "Graduate": {"Entry-Level", "Mid-Professional"},
    "Mid-Professional": {"Graduate", "Professional Individual Contributor", "Manager"},
    "Professional Individual Contributor": {"Mid-Professional", "Manager"},
    "Manager": {"Professional Individual Contributor", "Director", "Front Line Manager"},
    "Front Line Manager": {"Supervisor", "Manager"},
    "Supervisor": {"Front Line Manager"},
    "Director": {"Manager", "Executive"},
    "Executive": {"Director"},
}


class RetrievalCache:
    """
    Simple LRU cache for retrieval results.
    Keyed on a hash of the structured requirements so identical
    queries (common during evaluator replay) skip the full pipeline.
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._max_size = max_size

    def _make_key(self, state: dict) -> str:
        key_parts = [
            state.get("role", ""),
            "|".join(sorted(state.get("skills", []))),
            "|".join(sorted(state.get("job_level", []))),
            "|".join(sorted(state.get("languages", []))),
            "|".join(sorted(state.get("test_type_preferences", []))),
            "|".join(sorted(state.get("excluded_tests", []))),
        ]
        raw = "||".join(key_parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, state: dict) -> Optional[list[dict]]:
        key = self._make_key(state)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, state: dict, results: list[dict]) -> None:
        key = self._make_key(state)
        self._cache[key] = results
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


class RetrievalEngine:
    """
    Hybrid retrieval pipeline.

    Initialized once at startup with the loaded catalog and embedding index.
    The embedding model is also loaded once and reused for query encoding.
    """

    def __init__(self, catalog: CatalogStore):
        self.catalog = catalog
        self.cache = RetrievalCache()

        # load pre-computed embeddings
        logger.info("Loading embedding index from %s", EMBEDDINGS_PATH)
        data = np.load(EMBEDDINGS_PATH, allow_pickle=False)
        self._embeddings = data["embeddings"]  # shape: (N, dim)
        self._entity_ids = list(data["entity_ids"])  # list of entity_id strings

        # load the embedding model for query encoding
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        # build entity_id -> index mapping for fast lookup
        self._id_to_idx = {eid: i for i, eid in enumerate(self._entity_ids)}
        self._id_to_entry = {e.entity_id: e for e in catalog.entries}

        logger.info(
            "Retrieval engine ready: %d entries, %d embeddings",
            len(catalog.entries), len(self._entity_ids),
        )

    def retrieve(self, structured_state: dict) -> list[dict]:
        """
        Run the full retrieval pipeline and return top-K candidates
        as a list of dicts suitable for the LLM prompt context.
        """
        # step 0: cache check
        if CACHE_ENABLED:
            cached = self.cache.get(structured_state)
            if cached is not None:
                logger.info("Retrieval cache hit")
                return cached

        # step 1: build query signals
        query_text = self._build_query_text(structured_state)
        skill_keywords = [s.lower() for s in structured_state.get("skills", [])]
        required_levels = structured_state.get("job_level", [])
        required_languages = structured_state.get("languages", [])
        type_prefs = structured_state.get("test_type_preferences", [])
        excluded_names = set(
            n.lower() for n in structured_state.get("excluded_tests", [])
        )

        # step 2: semantic vector search
        semantic_scores = self._semantic_search(query_text)

        # step 3: keyword boost
        keyword_scores = self._keyword_search(skill_keywords)

        # merge both result sets (union by entity_id)
        candidates = self._merge_results(semantic_scores, keyword_scores)

        # step 4: business-rule hard filters
        candidates = self._apply_hard_filters(
            candidates, required_levels, required_languages, excluded_names
        )

        # step 5: weighted ranking
        scored = self._score_candidates(
            candidates, skill_keywords, required_levels,
            required_languages, type_prefs,
        )

        # sort by score descending, take top-K
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_k = scored[:RETRIEVAL_TOP_K]

        # step 6: OPQ32r default-inclusion heuristic
        top_k = self._maybe_add_opq32r(top_k, structured_state, excluded_names)

        # format for downstream consumption
        result = [self._format_candidate(c) for c in top_k]

        if CACHE_ENABLED:
            self.cache.put(structured_state, result)

        return result

    def _build_query_text(self, state: dict) -> str:
        """Construct a natural-language query for embedding."""
        parts = []
        if state.get("role"):
            parts.append(state["role"])
        if state.get("seniority"):
            parts.append(state["seniority"])
        for skill in state.get("skills", []):
            parts.append(skill)
        if state.get("industry"):
            parts.append(state["industry"])
        if state.get("additional_constraints"):
            parts.append(state["additional_constraints"])
        return " ".join(parts) if parts else "general assessment"

    def _semantic_search(self, query_text: str) -> dict[str, float]:
        """
        Encode the query and compute cosine similarity against all
        catalog embeddings. Returns {entity_id: similarity_score}.

        Since embeddings are pre-normalized, cosine similarity is
        just a dot product.
        """
        query_vec = self._model.encode(
            [query_text],
            normalize_embeddings=True,
        )  # shape: (1, dim)

        # dot product gives cosine similarity for normalized vectors
        similarities = np.dot(self._embeddings, query_vec.T).flatten()

        # take top-N candidates
        top_indices = np.argsort(similarities)[::-1][:SEMANTIC_CANDIDATES]

        scores = {}
        for idx in top_indices:
            eid = self._entity_ids[idx]
            scores[eid] = float(similarities[idx])

        return scores

    def _keyword_search(self, keywords: list[str]) -> dict[str, float]:
        """
        Scan catalog for entries whose name or description contains
        any of the skill keywords. Returns {entity_id: boost_score}.
        """
        if not keywords:
            return {}

        scores = {}
        for entry in self.catalog.entries:
            name_lower = entry.name.lower()
            desc_lower = entry.description.lower()
            match_count = 0
            for kw in keywords:
                if kw in name_lower or kw in desc_lower:
                    match_count += 1
            if match_count > 0:
                scores[entry.entity_id] = match_count / len(keywords)

        return scores

    def _merge_results(
        self,
        semantic: dict[str, float],
        keyword: dict[str, float],
    ) -> list[dict]:
        """Union the semantic and keyword result sets with both raw scores."""
        all_ids = set(semantic.keys()) | set(keyword.keys())
        merged = []
        for eid in all_ids:
            entry = self._id_to_entry.get(eid)
            if entry is None:
                continue
            merged.append({
                "entry": entry,
                "semantic_score": semantic.get(eid, 0.0),
                "keyword_score": keyword.get(eid, 0.0),
            })
        return merged

    def _apply_hard_filters(
        self,
        candidates: list[dict],
        required_levels: list[str],
        required_languages: list[str],
        excluded_names: set[str],
    ) -> list[dict]:
        """Drop candidates that violate hard constraints."""
        filtered = []
        for c in candidates:
            entry: CatalogEntry = c["entry"]

            # exclusion filter
            if entry.name.lower() in excluded_names:
                continue

            # language filter: penalize rather than drop outright
            # (some entries legitimately lack language metadata)
            c["language_penalty"] = False
            if required_languages:
                entry_langs = [lang.lower() for lang in entry.languages]
                if entry_langs:
                    has_match = any(
                        req.lower() in " ".join(entry_langs)
                        for req in required_languages
                    )
                    if not has_match:
                        c["language_penalty"] = True

            filtered.append(c)

        return filtered

    def _score_candidates(
        self,
        candidates: list[dict],
        skill_keywords: list[str],
        required_levels: list[str],
        required_languages: list[str],
        type_prefs: list[str],
    ) -> list[dict]:
        """
        Apply the weighted scoring function.

        Final Score =
            WEIGHT_SEMANTIC   x semantic_score
          + WEIGHT_SKILL_MATCH x skill_match_score
          + WEIGHT_JOB_LEVEL  x job_level_score
          + WEIGHT_LANGUAGE   x language_score
          + WEIGHT_TEST_CATEGORY x test_category_score
        """
        for c in candidates:
            entry: CatalogEntry = c["entry"]

            sem = c["semantic_score"]

            # skill match: fraction of keywords found in name + description
            if skill_keywords:
                text = (entry.name + " " + entry.description).lower()
                matches = sum(1 for kw in skill_keywords if kw in text)
                skill_score = matches / len(skill_keywords)
            else:
                skill_score = 0.5

            # job level match
            if required_levels:
                entry_levels = set(entry.job_levels)
                req_set = set(required_levels)
                if entry_levels & req_set:
                    level_score = 1.0
                elif any(
                    adj in entry_levels
                    for r in req_set
                    for adj in ADJACENT_LEVELS.get(r, set())
                ):
                    level_score = 0.5
                else:
                    level_score = 0.0
            else:
                level_score = 0.5

            # language match
            if required_languages:
                lang_score = 0.2 if c.get("language_penalty") else 1.0
            else:
                lang_score = 0.5

            # test category preference
            if type_prefs:
                entry_codes = set(entry.test_type.split(","))
                cat_score = 1.0 if entry_codes & set(type_prefs) else 0.3
            else:
                cat_score = 0.5

            c["score"] = (
                WEIGHT_SEMANTIC * sem
                + WEIGHT_SKILL_MATCH * skill_score
                + WEIGHT_JOB_LEVEL * level_score
                + WEIGHT_LANGUAGE * lang_score
                + WEIGHT_TEST_CATEGORY * cat_score
            )

        return candidates

    def _maybe_add_opq32r(
        self,
        candidates: list[dict],
        state: dict,
        excluded_names: set[str],
    ) -> list[dict]:
        """
        OPQ32r default-inclusion heuristic.

        For professional/managerial/technical roles where personality
        has not been explicitly declined, append OPQ32r if not already
        present. Skip for high-volume entry-level screening (C3 pattern).
        """
        opq_name = "Occupational Personality Questionnaire OPQ32r"

        if opq_name.lower() in excluded_names:
            return candidates
        if state.get("personality_required") is False:
            return candidates
        if state.get("volume_screening"):
            return candidates

        # check if already present
        for c in candidates:
            if c["entry"].name == opq_name:
                return candidates

        opq_entry = self.catalog.get_by_name(opq_name)
        if opq_entry:
            candidates.append({
                "entry": opq_entry,
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "score": 0.35,
            })

        return candidates

    @staticmethod
    def _format_candidate(c: dict) -> dict:
        """Format a scored candidate for the LLM prompt context."""
        entry: CatalogEntry = c["entry"]
        return {
            "name": entry.name,
            "url": entry.url,
            "test_type": entry.test_type,
            "duration": entry.duration,
            "job_levels": entry.job_levels,
            "languages": entry.languages,
            "description": entry.description,
            "keys": entry.keys,
            "retrieval_score": round(c["score"], 3),
        }
