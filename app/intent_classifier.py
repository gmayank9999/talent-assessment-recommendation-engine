"""
Intent classifier for the conversation state machine.

Classifies each incoming turn into exactly one of six states:
    CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE, CONFIRM

Uses a rule-based approach first (covers most cases deterministically),
falling back to heuristic classification when rules are ambiguous.
This avoids a second LLM call for intent detection.
"""

import re
import logging
from enum import Enum

from app.conversation_parser import ParsedConversation

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    CLARIFY = "CLARIFY"
    RECOMMEND = "RECOMMEND"
    REFINE = "REFINE"
    COMPARE = "COMPARE"
    REFUSE = "REFUSE"
    CONFIRM = "CONFIRM"


# prompt injection patterns -- match these before any other classification
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are", re.IGNORECASE),
    re.compile(r"disregard\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+your\s+prompt", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(your\s+)?instructions", re.IGNORECASE),
]

# off-topic signals that should trigger REFUSE
_OFFTOPIC_PATTERNS = [
    re.compile(r"\b(legal|legally)\s+(require|obligation|compliance|advice)\b", re.IGNORECASE),
    re.compile(r"\bare\s+we\s+(legally|required)\b", re.IGNORECASE),
    re.compile(r"\b(immigration|visa|salary|compensation)\s+(advice|help)\b", re.IGNORECASE),
]

# compare signals
_COMPARE_PATTERNS = [
    re.compile(r"(?:what(?:'s| is) the )?differ(?:ence|ent)\s+between", re.IGNORECASE),
    re.compile(r"compare\s+(?:the\s+)?(.+?)\s+(?:and|vs|with|to)", re.IGNORECASE),
    re.compile(r"how\s+(?:does|do|is)\s+(.+?)\s+differ\s+from", re.IGNORECASE),
]

# confirmation signals
_CONFIRM_PATTERNS = [
    re.compile(r"^(?:perfect|confirmed?|locking\s+it\s+in|that(?:'s)?\s+(?:good|great|what\s+we\s+need)|keep\s+(?:the|it)|finalized?)[\.\!\s]*$", re.IGNORECASE),
    re.compile(r"(?:that\s+works|looks?\s+good|sounds?\s+good)\.?\s*(?:thanks?|thank\s+you)?\.?\s*$", re.IGNORECASE),
]

# modification signals that point to REFINE
_REFINE_PATTERNS = [
    re.compile(r"\b(add|include|drop|remove|exclude|replace|swap)\b", re.IGNORECASE),
    re.compile(r"\b(keep.+(?:as.is|unchanged)|final\s+list)\b", re.IGNORECASE),
]


def _has_enough_context(state: dict) -> bool:
    """
    Check if the accumulated requirements have enough signal
    to produce a useful recommendation.

    Sufficient means: we have at least a role/domain OR explicit skills
    AND some indication of seniority or job level.
    """
    has_role = bool(state.get("role"))
    has_skills = len(state.get("skills", [])) >= 1
    has_seniority = bool(state.get("seniority")) or len(state.get("job_level", [])) >= 1
    has_domain = bool(state.get("industry"))

    # a rich JD or explicit skill list can be enough even without seniority
    if has_skills and len(state.get("skills", [])) >= 3:
        return True

    # role + seniority is the standard minimum
    if has_role and has_seniority:
        return True

    # domain + seniority also works (e.g., "senior leadership")
    if has_domain and has_seniority:
        return True

    # explicit test type requests are enough
    if state.get("test_type_preferences"):
        return True

    # volume screening has enough context from the role alone
    if state.get("volume_screening") and has_role:
        return True

    return False


class IntentClassifier:
    """
    Determines the conversation intent for the current turn.

    Decision order:
    1. Check for prompt injection -> REFUSE
    2. Check for off-topic content -> REFUSE
    3. Check for compare requests -> COMPARE
    4. Check for confirmation -> CONFIRM
    5. Check for add/remove with existing shortlist -> REFINE
    6. Has enough context + no shortlist -> RECOMMEND
    7. Not enough context -> CLARIFY
    8. Has shortlist + no modification -> echo shortlist (RECOMMEND)
    """

    def classify(
        self,
        latest_user_message: str,
        structured_state: dict,
        parsed: ParsedConversation,
    ) -> Intent:

        text = latest_user_message.strip()

        # 1. prompt injection check -- reject immediately
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                logger.warning("Prompt injection detected")
                return Intent.REFUSE

        # 2. off-topic check
        for pat in _OFFTOPIC_PATTERNS:
            if pat.search(text):
                logger.info("Off-topic content detected")
                return Intent.REFUSE

        # 3. compare check -- "what's the difference between X and Y?"
        for pat in _COMPARE_PATTERNS:
            if pat.search(text):
                return Intent.COMPARE

        # 4. confirmation check -- "Perfect, that's what we need."
        # only valid if there is already a shortlist to confirm
        if parsed.has_prior_shortlist:
            for pat in _CONFIRM_PATTERNS:
                if pat.search(text):
                    return Intent.CONFIRM

        # 5. refine check -- add/remove/drop/swap with existing shortlist
        if parsed.has_prior_shortlist:
            has_modification = bool(
                parsed.explicit_additions or parsed.explicit_removals
            )
            for pat in _REFINE_PATTERNS:
                if pat.search(text):
                    has_modification = True
                    break

            if has_modification:
                return Intent.REFINE

        # 6-8. RECOMMEND vs CLARIFY based on accumulated context
        enough = _has_enough_context(structured_state)
        forced = parsed.clarification_turns_used >= 2

        if enough or forced:
            return Intent.RECOMMEND
        else:
            return Intent.CLARIFY
