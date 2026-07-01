"""
Unit tests for the ConversationParser.

Tests cover:
    - shortlist extraction from assistant messages
    - clarification turn counting
    - explicit add/remove parsing
    - conjunction splitting ("Add AWS and Docker. Drop REST")
    - edge cases: empty history, no prior recommendations
"""

import json
import pytest
from app.conversation_parser import ConversationParser, _split_items


@pytest.fixture
def parser():
    return ConversationParser()


def _assistant_msg(reply: str, recs=None, eoc=False):
    """Helper to build an assistant message in the stored format."""
    body = {
        "reply": reply,
        "recommendations": recs,
        "end_of_conversation": eoc,
    }
    return {"role": "assistant", "content": json.dumps(body)}


def _user_msg(text: str):
    return {"role": "user", "content": text}


class TestShortlistExtraction:

    def test_finds_last_recommendation(self, parser):
        messages = [
            _user_msg("I need an assessment"),
            _assistant_msg("Sure", recs=[
                {"name": "Test A", "url": "https://www.shl.com/a/", "test_type": "K"}
            ]),
            _user_msg("Add more"),
            _assistant_msg("Added", recs=[
                {"name": "Test A", "url": "https://www.shl.com/a/", "test_type": "K"},
                {"name": "Test B", "url": "https://www.shl.com/b/", "test_type": "P"},
            ]),
        ]
        result = parser.parse(messages)
        assert result.has_prior_shortlist is True
        assert len(result.current_shortlist) == 2
        assert result.current_shortlist[1]["name"] == "Test B"

    def test_no_recommendations_in_history(self, parser):
        messages = [
            _user_msg("Hello"),
            _assistant_msg("What role?", recs=None),
        ]
        result = parser.parse(messages)
        assert result.has_prior_shortlist is False
        assert result.current_shortlist == []

    def test_empty_messages(self, parser):
        result = parser.parse([])
        assert result.has_prior_shortlist is False
        assert result.clarification_turns_used == 0


class TestClarificationCount:

    def test_counts_null_rec_turns(self, parser):
        messages = [
            _user_msg("I need help"),
            _assistant_msg("What role?", recs=None),
            _user_msg("Engineering"),
            _assistant_msg("What level?", recs=None),
        ]
        result = parser.parse(messages)
        assert result.clarification_turns_used == 2

    def test_resets_after_recommendation(self, parser):
        messages = [
            _user_msg("Java developer"),
            _assistant_msg("Here you go", recs=[
                {"name": "Java 8", "url": "https://www.shl.com/java/", "test_type": "K"}
            ]),
            _user_msg("What about Angular?"),
            _assistant_msg("Let me check", recs=None),
        ]
        result = parser.parse(messages)
        assert result.clarification_turns_used == 1


class TestAddRemoveExtraction:

    def test_simple_add(self, parser):
        messages = [_user_msg("Add AWS")]
        result = parser.parse(messages)
        assert "AWS" in result.explicit_additions

    def test_simple_drop(self, parser):
        messages = [_user_msg("Drop REST")]
        result = parser.parse(messages)
        assert "REST" in result.explicit_removals

    def test_compound_instruction(self, parser):
        """Trace C9 turn 4 pattern: 'Add AWS and Docker. Drop REST'"""
        messages = [_user_msg("Add AWS and Docker. Drop REST")]
        result = parser.parse(messages)
        assert "AWS" in result.explicit_additions
        assert "Docker" in result.explicit_additions
        assert "REST" in result.explicit_removals


class TestSplitItems:

    def test_and_conjunction(self):
        assert _split_items("AWS and Docker") == ["AWS", "Docker"]

    def test_comma_separated(self):
        assert _split_items("Java, Spring, SQL") == ["Java", "Spring", "SQL"]

    def test_single_item(self):
        assert _split_items("REST") == ["REST"]

    def test_trailing_punctuation(self):
        assert _split_items("the OPQ.") == ["the OPQ"]
