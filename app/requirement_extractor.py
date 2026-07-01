"""
Requirement extractor.

Takes the full message history and parsed conversation state, then
produces a structured requirement object that captures:
  - role, seniority, skills, languages, industry
  - test type preferences, safety/compliance context
  - volume screening and development context flags

This uses a single LLM call with a focused extraction prompt to pull
structured data from conversational text. The ParsedConversation fields
(current_shortlist, clarification_turns_used, explicit additions/removals)
are passed through directly -- not re-derived here.
"""

import json
import logging
from typing import Optional

from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from app.conversation_parser import ParsedConversation

logger = logging.getLogger(__name__)

# the extraction prompt asks the LLM to read the conversation and
# return structured fields -- nothing more
_EXTRACTION_PROMPT = """Read the conversation below and extract the hiring requirements mentioned.
Return a JSON object with these fields only. Use null for fields not mentioned.

{
  "role": "job title or role being hired for",
  "seniority": "entry-level / graduate / mid-level / senior / director / executive",
  "skills": ["list of technical skills or competencies mentioned"],
  "languages": ["assessment languages needed, e.g. Latin American Spanish"],
  "industry": "industry or domain if mentioned",
  "test_type_preferences": ["K/P/A/B/S/C/D codes if user explicitly asked for certain test types"],
  "personality_required": true/false/null,
  "ability_required": true/false/null,
  "knowledge_required": true/false/null,
  "simulation_required": true/false/null,
  "safety_critical": false,
  "bilingual_required": false,
  "volume_screening": false,
  "development_context": false,
  "additional_constraints": "any other constraints not captured above"
}

Only return valid JSON, nothing else."""


def _build_conversation_text(messages: list[dict]) -> str:
    """Format messages into a readable conversation transcript."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _map_seniority_to_job_levels(seniority: Optional[str]) -> list[str]:
    """
    Map a free-text seniority description to the catalog's job_level vocabulary.
    Handles common variations like "5+ years", "senior IC", "CXO", etc.
    """
    if not seniority:
        return []

    s = seniority.lower()
    levels = []

    # direct matches
    level_map = {
        "entry": ["Entry-Level"],
        "graduate": ["Graduate"],
        "junior": ["Entry-Level", "Graduate"],
        "mid": ["Mid-Professional", "Professional Individual Contributor"],
        "senior": ["Mid-Professional", "Professional Individual Contributor"],
        "lead": ["Manager", "Front Line Manager"],
        "manager": ["Manager", "Front Line Manager"],
        "director": ["Director"],
        "executive": ["Executive", "Director"],
        "cxo": ["Executive"],
    }

    for keyword, mapped in level_map.items():
        if keyword in s:
            levels.extend(mapped)

    # experience-based heuristics
    import re
    years_match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)", s)
    if years_match:
        years = int(years_match.group(1))
        if years <= 2:
            levels.extend(["Entry-Level", "Graduate"])
        elif years <= 5:
            levels.extend(["Mid-Professional", "Professional Individual Contributor"])
        elif years <= 10:
            levels.extend(["Mid-Professional", "Professional Individual Contributor"])
        else:
            levels.extend(["Director", "Executive"])

    # deduplicate while preserving order
    seen = set()
    unique = []
    for level in levels:
        if level not in seen:
            seen.add(level)
            unique.append(level)

    return unique if unique else ["General Population"]


class RequirementExtractor:
    """
    Extracts structured hiring requirements from conversation history.

    Uses one lightweight LLM call for semantic extraction (role, skills, etc.)
    and merges the result with deterministic fields from ParsedConversation.
    """

    def __init__(self):
        self._client = Groq(api_key=GROQ_API_KEY)

    def extract(
        self,
        messages: list[dict],
        parsed: ParsedConversation,
    ) -> dict:
        """
        Produce the full structured requirement object.

        The LLM extracts semantic fields (role, skills, seniority, etc.)
        and the deterministic fields from ParsedConversation are merged in.
        """
        conversation_text = _build_conversation_text(messages)

        # call the LLM for semantic extraction
        extracted = self._llm_extract(conversation_text)

        # map seniority to catalog job_level vocabulary
        job_levels = _map_seniority_to_job_levels(extracted.get("seniority"))

        # build the final structured state
        state = {
            "role": extracted.get("role"),
            "seniority": extracted.get("seniority"),
            "job_level": job_levels,
            "skills": extracted.get("skills") or [],
            "must_include_tests": [],
            "excluded_tests": [
                name for name in parsed.explicit_removals
            ],
            "languages": extracted.get("languages") or [],
            "industry": extracted.get("industry"),
            "test_type_preferences": extracted.get("test_type_preferences") or [],
            "personality_required": extracted.get("personality_required"),
            "ability_required": extracted.get("ability_required"),
            "knowledge_required": extracted.get("knowledge_required"),
            "simulation_required": extracted.get("simulation_required"),
            "safety_critical": extracted.get("safety_critical", False),
            "bilingual_required": extracted.get("bilingual_required", False),
            "volume_screening": extracted.get("volume_screening", False),
            "development_context": extracted.get("development_context", False),
            "current_shortlist": parsed.current_shortlist,
            "additional_constraints": extracted.get("additional_constraints"),
            "clarification_turns_used": parsed.clarification_turns_used,
        }

        return state

    def _llm_extract(self, conversation_text: str) -> dict:
        """
        Call the LLM with the extraction prompt.
        Returns the parsed JSON or empty dict on failure.
        """
        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _EXTRACTION_PROMPT},
                    {"role": "user", "content": conversation_text},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=800,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            return {}

        except Exception as e:
            logger.error("Requirement extraction LLM call failed: %s", e)
            return {}
