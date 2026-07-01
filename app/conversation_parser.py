"""
Conversation parser -- deterministic, no LLM.

Runs on every /chat call before anything else. Extracts structured facts
from the raw messages[] history that the rest of the system depends on:

1. The current shortlist (from the last assistant turn with recommendations)
2. How many consecutive clarification turns have passed
3. Explicit add/remove instructions from user messages
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedConversation:
    # items from the most recent assistant recommendation turn
    current_shortlist: list[dict] = field(default_factory=list)

    # count of consecutive assistant turns with no recommendations
    clarification_turns_used: int = 0

    # assessment/tech names the user explicitly asked to add
    explicit_additions: list[str] = field(default_factory=list)

    # assessment names the user explicitly asked to remove
    explicit_removals: list[str] = field(default_factory=list)

    # whether any prior recommendation turn exists in history
    has_prior_shortlist: bool = False

    # index of the last assistant turn that carried recommendations
    last_recommendation_turn_index: Optional[int] = None


# patterns for detecting add/remove intent in user messages
_ADD_PATTERNS = [
    re.compile(r"\badd\s+(.+)", re.IGNORECASE),
    re.compile(r"\binclude\s+(.+)", re.IGNORECASE),
    re.compile(r"\balso\s+add\s+(.+)", re.IGNORECASE),
]

_REMOVE_PATTERNS = [
    re.compile(r"\bdrop\s+(.+)", re.IGNORECASE),
    re.compile(r"\bremove\s+(.+)", re.IGNORECASE),
    re.compile(r"\bexclude\s+(.+)", re.IGNORECASE),
    re.compile(r"\bwithout\s+(.+)", re.IGNORECASE),
]

# words that separate multiple items in add/remove instructions
_SPLIT_PATTERN = re.compile(r"\s+and\s+|\s*,\s*|\s*&\s*", re.IGNORECASE)

# stop phrases that should terminate a match
# e.g. "Add AWS and Docker. Drop REST" -- "." separates add from drop
_STOP_PHRASES = re.compile(
    r"\.\s*(?:drop|remove|but|however|exclude|add|include)",
    re.IGNORECASE,
)


def _split_items(raw: str) -> list[str]:
    """
    Split a matched group into individual item names.
    Handles conjunctions: "AWS and Docker" -> ["AWS", "Docker"]
    Also handles: "AWS, Docker, and Kubernetes"
    """
    # trim trailing punctuation and stop-phrase tails
    stop = _STOP_PHRASES.search(raw)
    if stop:
        raw = raw[:stop.start()]

    # strip trailing punctuation
    raw = raw.rstrip(".,;!?")

    parts = _SPLIT_PATTERN.split(raw)
    return [p.strip() for p in parts if p.strip()]


def _extract_modifications(text: str) -> tuple[list[str], list[str]]:
    """
    Pull explicit add/remove instructions from a single user message.
    Returns (additions, removals).
    """
    additions = []
    removals = []

    for pat in _ADD_PATTERNS:
        for m in pat.finditer(text):
            additions.extend(_split_items(m.group(1)))

    for pat in _REMOVE_PATTERNS:
        for m in pat.finditer(text):
            removals.extend(_split_items(m.group(1)))

    return additions, removals


class ConversationParser:
    """
    Stateless parser that produces a ParsedConversation from raw messages.

    The output of this parser is passed into the requirement extractor
    and intent classifier. The REFINE behavior relies on
    current_shortlist from this parser, not from re-parsing inside the LLM.
    """

    def parse(self, messages: list[dict]) -> ParsedConversation:
        result = ParsedConversation()

        # step 1: find the last assistant message with a non-null shortlist
        # scan backwards so we find the most recent one first
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "assistant":
                continue
            recs = self._extract_recs(msg.get("content", ""))
            if recs is not None and len(recs) > 0:
                result.current_shortlist = recs
                result.has_prior_shortlist = True
                result.last_recommendation_turn_index = i
                break

        # step 2: count clarification turns since the last recommendation
        start_idx = (result.last_recommendation_turn_index or -1) + 1
        null_run = 0
        for msg in messages[start_idx:]:
            if msg.get("role") != "assistant":
                continue
            recs = self._extract_recs(msg.get("content", ""))
            if recs is None or len(recs) == 0:
                null_run += 1
            else:
                null_run = 0
        result.clarification_turns_used = null_run

        # step 3: extract explicit add/remove from the latest user message only
        # (older user messages are already reflected in the current shortlist)
        for msg in reversed(messages):
            if msg.get("role") == "user":
                adds, removes = _extract_modifications(msg["content"])
                result.explicit_additions = adds
                result.explicit_removals = removes
                break

        return result

    @staticmethod
    def _extract_recs(content: str) -> Optional[list[dict]]:
        """
        Try to parse recommendations from an assistant message.
        Assistant messages can be either raw JSON or contain a JSON body.
        """
        if not content:
            return None
        try:
            parsed = json.loads(content)
            # could be the full ChatResponse dict
            if isinstance(parsed, dict):
                return parsed.get("recommendations")
            return None
        except (json.JSONDecodeError, TypeError):
            return None
