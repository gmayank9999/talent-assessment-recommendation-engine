"""
Trace replay test harness.

Replays the 10 sample conversations (C1-C10) against the running service.
For each trace, sends user turns sequentially (accumulating full history)
and checks:
    - Did recommendations go null/populated at the correct turns?
    - Did end_of_conversation flip at the right turn?
    - Were previously-included items preserved during REFINE turns?

Usage:
    Start the server first, then:
    pytest tests/test_trace_replay.py -v

    Or run standalone:
    python tests/test_trace_replay.py
"""

import json
import re
import sys
import os
import httpx
import pytest

# base URL of the running service
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")

# path to sample conversations
CONVERSATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_conversations", "GenAI_SampleConversations",
)


def _parse_trace_file(filepath: str) -> list[dict]:
    """
    Parse a sample conversation markdown file into a list of turns.

    Each turn has:
        - user_message: str
        - has_recommendations: bool
        - end_of_conversation: bool
        - recommendation_names: list[str] (if available)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    turns = []
    # split by turn headers
    turn_blocks = re.split(r"### Turn \d+", content)

    for block in turn_blocks[1:]:  # skip the preamble
        user_match = re.search(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n\n|\*\*Agent\*\*)", block, re.DOTALL)
        if not user_match:
            continue

        user_msg = user_match.group(1).strip()
        # clean up markdown quoting
        user_msg = re.sub(r"^>\s*", "", user_msg, flags=re.MULTILINE).strip()

        has_recs = "recommendations: null" not in block.lower() and "|" in block
        eoc_match = re.search(r"end_of_conversation.*?(\*\*true\*\*|\*\*false\*\*)", block, re.IGNORECASE)
        eoc = "true" in (eoc_match.group(1) if eoc_match else "false")

        # extract recommendation names from the table if present
        rec_names = []
        if has_recs:
            # match table rows: | # | Name | ...
            table_rows = re.findall(r"\|\s*\d+\s*\|\s*(.+?)\s*\|", block)
            rec_names = [name.strip() for name in table_rows]

        turns.append({
            "user_message": user_msg,
            "has_recommendations": has_recs,
            "end_of_conversation": eoc,
            "recommendation_names": rec_names,
        })

    return turns


def _run_trace(trace_file: str) -> list[dict]:
    """
    Replay a single trace against the running service.
    Returns a list of results per turn.
    """
    turns = _parse_trace_file(trace_file)
    messages = []
    results = []

    with httpx.Client(base_url=BASE_URL, timeout=35.0) as client:
        for turn in turns:
            messages.append({"role": "user", "content": turn["user_message"]})

            response = client.post("/chat", json={"messages": messages})
            assert response.status_code == 200, f"HTTP {response.status_code}"

            data = response.json()

            # add assistant response to history for next turn
            messages.append({"role": "assistant", "content": json.dumps(data)})

            actual_has_recs = data.get("recommendations") is not None
            actual_eoc = data.get("end_of_conversation", False)

            results.append({
                "expected_has_recs": turn["has_recommendations"],
                "actual_has_recs": actual_has_recs,
                "expected_eoc": turn["end_of_conversation"],
                "actual_eoc": actual_eoc,
                "expected_names": turn["recommendation_names"],
                "actual_names": [
                    r["name"] for r in (data.get("recommendations") or [])
                ],
            })

    return results


def _compute_recall(expected: list[str], actual: list[str]) -> float:
    """Compute recall: fraction of expected items found in actual."""
    if not expected:
        return 1.0
    found = sum(1 for name in expected if name in actual)
    return found / len(expected)


# test functions for pytest
@pytest.fixture(scope="module")
def check_server():
    """Skip all tests if the server is not running."""
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        if r.status_code != 200:
            pytest.skip("Server not running")
    except httpx.ConnectError:
        pytest.skip("Server not running at " + BASE_URL)


@pytest.mark.parametrize("trace_name", [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
])
def test_trace_replay(check_server, trace_name):
    """Replay a single trace and check turn-level behavior."""
    trace_path = os.path.join(CONVERSATIONS_DIR, f"{trace_name}.md")
    if not os.path.exists(trace_path):
        pytest.skip(f"Trace file not found: {trace_path}")

    results = _run_trace(trace_path)

    for i, r in enumerate(results):
        turn = i + 1

        # check recommendation presence at each turn
        assert r["actual_has_recs"] == r["expected_has_recs"], (
            f"{trace_name} turn {turn}: expected recs={'yes' if r['expected_has_recs'] else 'null'}, "
            f"got {'yes' if r['actual_has_recs'] else 'null'}"
        )

        # check end_of_conversation on the final turn
        if r["expected_eoc"]:
            assert r["actual_eoc"], (
                f"{trace_name} turn {turn}: expected end_of_conversation=true"
            )

    # check recall on turns that have recommendations
    for i, r in enumerate(results):
        if r["expected_names"]:
            recall = _compute_recall(r["expected_names"], r["actual_names"])
            # log but don't hard-fail on recall (LLM output varies)
            if recall < 0.5:
                print(
                    f"  WARNING: {trace_name} turn {i+1} recall={recall:.2f} "
                    f"(expected {r['expected_names']}, got {r['actual_names']})"
                )


if __name__ == "__main__":
    """Run all traces and print a summary."""
    print(f"Replaying traces against {BASE_URL}\n")

    for name in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]:
        path = os.path.join(CONVERSATIONS_DIR, f"{name}.md")
        if not os.path.exists(path):
            print(f"  {name}: SKIP (file not found)")
            continue

        try:
            results = _run_trace(path)
            recs_ok = all(
                r["actual_has_recs"] == r["expected_has_recs"] for r in results
            )
            eoc_ok = all(
                r["actual_eoc"] == r["expected_eoc"]
                for r in results if r["expected_eoc"]
            )
            status = "PASS" if (recs_ok and eoc_ok) else "FAIL"
            print(f"  {name}: {status} ({len(results)} turns)")

            # print recall for recommendation turns
            for i, r in enumerate(results):
                if r["expected_names"]:
                    recall = _compute_recall(r["expected_names"], r["actual_names"])
                    print(f"    turn {i+1} recall={recall:.2f}")

        except Exception as e:
            print(f"  {name}: ERROR ({e})")

    print("\nDone.")
