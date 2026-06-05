"""Deterministic terminal bot-chatter suppression for the Discord gateway.

Issue #172 (drj0717/neuromancer#172).

Agents coordinating over Discord may post pure control / acknowledgement
chatter — ``ack``, ``final``, ``status``, ``no-op``, ``standing down``,
``no further action``, ``closed``. When ``DISCORD_ALLOW_BOTS`` permits other
bots (``mentions`` or ``all``), dispatching those *terminal* posts to the LLM
wakes a peer agent, which then posts its own ack — a reciprocal-ACK / visible
loop. This module classifies a bot message's protocol *kind* so the gateway
can drop terminal chatter *before* LLM dispatch, while still routing
*actionable* bot messages (``request`` / ``handoff`` / ``requires-ack``).

Determinism rule (the #174 lesson, generalized): classification keys ONLY on
an explicit ``kind:`` protocol marker that the sending agent sets — never on
free prose. A post such as ``"standing down — no further action on the
BLOCKED deploy"`` contains actionable words and must fail safe toward
*routing*, never toward silently dropping a real request. No marker, an
unrecognized kind, or a marker buried mid-prose → not terminal → route.

The vocabulary mirrors the agent-comm protocol
(``agent-comm/agent_comm/protocol.py::classify`` in Neuromancer): the kinds
for which ``actionable=False`` plus the explicit terminal kinds named in the
issue acceptance criteria.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["classify_bot_message", "is_terminal_bot_message", "TERMINAL_KINDS"]

# Terminal (non-actionable) kinds. Normalized form: lowercase, hyphenated.
# ``fyi`` / ``notification`` come from agent-comm's actionable=False set; the
# remainder are the explicit terminal kinds named in the #172 acceptance.
TERMINAL_KINDS = frozenset(
    {
        "ack",
        "final",
        "status",
        "no-op",
        "standing-down",
        "no-further-action",
        "closed",
        "fyi",
        "notification",
    }
)

# The marker must be *leading* (the protocol form), optionally preceded by
# Discord mention tokens — under ``DISCORD_ALLOW_BOTS=mentions`` a terminal
# post @mentions us, so ``kind:`` trails the mention. Only the first
# identifier token after ``kind:`` / ``kind=`` is captured; trailing
# human-readable prose on the same line is ignored.
_MENTION_PREFIX = r"(?:<@[!&]?\d+>\s*)*"
_KIND_MARKER = re.compile(
    rf"^\s*{_MENTION_PREFIX}kind\s*[:=]\s*['\"]?([A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)

# Alias normalization for spelling variants of the same kind.
_ALIASES = {
    "noop": "no-op",
}


def _normalize(kind: str) -> str:
    normalized = kind.strip().lower().replace("_", "-")
    return _ALIASES.get(normalized, normalized)


def classify_bot_message(content: Optional[str]) -> Optional[str]:
    """Return the normalized protocol ``kind`` of *content*, or ``None``.

    ``None`` means no explicit leading ``kind:`` marker was found — the
    message is treated as non-terminal (route) by every caller.
    """
    if not content:
        return None
    match = _KIND_MARKER.match(content)
    if not match:
        return None
    return _normalize(match.group(1))


def is_terminal_bot_message(content: Optional[str]) -> bool:
    """True iff *content* carries an explicit terminal-kind marker.

    Fails safe toward route: ambiguous prose, missing markers, and unknown
    kinds all return False so a real request is never dropped.
    """
    return classify_bot_message(content) in TERMINAL_KINDS
