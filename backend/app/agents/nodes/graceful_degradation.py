"""
Node 8 — GracefulDegradation

Runs when retrieval_attempts >= 2 AND retrieval_passed = False.

Sets confidence_tier and system_note so response_generation_node
produces an honest, qualified answer with a mandatory disclaimer.
Never returns an empty response to the user.
"""
import logging

from app.agents.state import LexFindState

logger = logging.getLogger(__name__)

_NOTES = {
    "too_specific": (
        "Note: The database did not contain an exact match for your query. "
        "The following answer is based on general legal principles and available related cases."
    ),
    "wrong_domain": (
        "Note: This topic may be outside the scope of the current case corpus. "
        "The response draws on general legal knowledge and may not be grounded in specific retrieved cases."
    ),
    "no_coverage": (
        "Note: This specific topic does not appear to be covered in the current corpus of Supreme Court cases. "
        "The following is based on general Indian legal principles."
    ),
}


def graceful_degradation_node(state: LexFindState) -> LexFindState:
    """
    Set confidence tier and inject system note based on grader failure reason.
    Response generation will use system_note to prepend the required disclaimer.
    """
    reason = state.get("grader_reason") or "no_coverage"

    logger.warning(
        "GracefulDegradation triggered after %d attempts, reason=%s",
        state.get("retrieval_attempts", 2), reason,
    )

    system_note = _NOTES.get(reason, _NOTES["no_coverage"])

    return {
        **state,
        "confidence_tier": "degraded",
        "system_note":     system_note,
    }
