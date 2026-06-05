"""Tests for Discord terminal bot-chatter suppression (issue #172).

When ``DISCORD_ALLOW_BOTS`` permits other bots (``mentions`` or ``all``), a
bot-authored *terminal* control message — ``ack`` / ``final`` / ``status`` /
``no-op`` / ``standing-down`` / ``no-further-action`` / ``closed`` — must be
dropped before LLM dispatch so peers do not cascade reciprocal ACKs.
*Actionable* bot messages (``request`` / ``handoff`` / ``requires-ack``) must
still route, human terminal chatter must still route, and ``allow_bots=none``
behaviour must be unchanged.

Two layers are covered:

1. The pure classifier ``classify_bot_message`` / ``is_terminal_bot_message``
   in ``plugins.platforms.discord.terminal_filter`` — full vocabulary,
   mention-prefixed markers, and the deterministic fail-safe-toward-route
   rule for ambiguous prose.
2. The ``on_message`` bot gate, replicated faithfully (the same pattern used
   by ``test_discord_bot_filter`` for the permit gate) and driven through the
   *real* classifier, asserting the accept/drop dispatch decision.
"""

import unittest
from unittest.mock import MagicMock

from plugins.platforms.discord.terminal_filter import (
    classify_bot_message,
    is_terminal_bot_message,
)


class TestClassifier(unittest.TestCase):
    """Unit tests for the pure kind classifier."""

    def test_terminal_kinds_classified(self):
        for marker, expected in [
            ("kind:ack", "ack"),
            ("kind:final", "final"),
            ("kind:status", "status"),
            ("kind:no-op", "no-op"),
            ("kind:no_op", "no-op"),
            ("kind:noop", "no-op"),
            ("kind:standing-down", "standing-down"),
            ("kind:no-further-action", "no-further-action"),
            ("kind:closed", "closed"),
            ("kind:fyi", "fyi"),
            ("kind:notification", "notification"),
        ]:
            with self.subTest(marker=marker):
                self.assertEqual(classify_bot_message(marker), expected)
                self.assertTrue(is_terminal_bot_message(marker))

    def test_actionable_kinds_not_terminal(self):
        for marker in [
            "kind:request",
            "kind:handoff",
            "kind:requires-ack",
            "kind:requires_ack",
        ]:
            with self.subTest(marker=marker):
                self.assertFalse(is_terminal_bot_message(marker))

    def test_marker_is_case_insensitive(self):
        self.assertEqual(classify_bot_message("KIND:ACK"), "ack")
        self.assertEqual(classify_bot_message("Kind: Final"), "final")

    def test_marker_accepts_equals_and_whitespace(self):
        self.assertEqual(classify_bot_message("kind = status"), "status")
        self.assertEqual(classify_bot_message("kind:  ack"), "ack")

    def test_quoted_marker_value(self):
        self.assertEqual(classify_bot_message('kind:"ack"'), "ack")
        self.assertEqual(classify_bot_message("kind:'final'"), "final")

    def test_marker_after_leading_mention(self):
        # Under allow_bots=mentions a terminal post @mentions us; the kind
        # marker then trails the mention token. It must still be detected.
        self.assertTrue(is_terminal_bot_message("<@123456789012345678> kind:ack done"))
        self.assertTrue(is_terminal_bot_message("<@!123> <@&456> kind:final"))

    def test_marker_followed_by_prose_uses_only_the_token(self):
        # The kind token then human text on the same line.
        self.assertEqual(classify_bot_message("kind:ack received, proceeding"), "ack")
        self.assertTrue(is_terminal_bot_message("kind:ack received, proceeding"))

    def test_ambiguous_prose_without_marker_routes(self):
        # The #174 lesson: never infer terminal from prose. These contain
        # terminal *words* but no explicit marker → must NOT be terminal.
        for prose in [
            "standing down — no further action on the BLOCKED deploy",
            "ack",
            "final answer below",
            "status: everything is closed and done",
            "no-op for now, but please pick up the request",
        ]:
            with self.subTest(prose=prose):
                self.assertIsNone(classify_bot_message(prose))
                self.assertFalse(is_terminal_bot_message(prose))

    def test_non_leading_marker_is_ignored(self):
        # A marker buried mid-prose is not the protocol form and must not
        # cause a real request to be dropped (fail safe toward route).
        self.assertIsNone(classify_bot_message("please send kind:ack when you finish the build"))
        self.assertFalse(is_terminal_bot_message("please send kind:ack when you finish the build"))

    def test_unknown_kind_routes(self):
        self.assertEqual(classify_bot_message("kind:banana"), "banana")
        self.assertFalse(is_terminal_bot_message("kind:banana"))

    def test_empty_and_none(self):
        self.assertIsNone(classify_bot_message(""))
        self.assertIsNone(classify_bot_message(None))
        self.assertFalse(is_terminal_bot_message(""))
        self.assertFalse(is_terminal_bot_message(None))


def _make_author(*, bot=False, is_self=False):
    author = MagicMock()
    author.bot = bot
    author.id = 99999 if is_self else 12345
    author.name = "TestBot" if bot else "TestUser"
    author.display_name = author.name
    return author


class TestDispatchGate(unittest.TestCase):
    """Replicate the on_message bot gate and assert the dispatch decision.

    Mirrors ``plugins/platforms/discord/adapter.py::on_message`` for the
    ``author.bot`` branch: permit (DISCORD_ALLOW_BOTS) then the terminal
    guard. Returns True when the message would reach ``_handle_message``
    (i.e. route to the LLM), False when it is dropped before dispatch.
    """

    def _would_dispatch(self, *, content, bot, allow_bots, mentioned=False, client_user=None):
        client_user = client_user or _make_author(is_self=True)
        message = MagicMock()
        message.author = _make_author(bot=bot)
        message.content = content
        message.mentions = [client_user] if mentioned else []

        # Own-message guard (always ignored) is handled earlier; not modelled.
        if getattr(message.author, "bot", False):
            allow = allow_bots.lower().strip()
            if allow == "none":
                return False
            elif allow == "mentions":
                if client_user not in message.mentions:
                    return False
            # "all" falls through; bot is permitted.
            # Terminal-chatter guard (issue #172): real classifier.
            if is_terminal_bot_message(message.content):
                return False
        # Non-bot path: terminal guard is never reached (bot-only).
        return True

    # ── allow_bots=all ───────────────────────────────────────────────
    def test_bot_terminal_dropped_under_all(self):
        for content in ["kind:ack", "kind:final", "kind:status", "kind:no-op",
                        "kind:standing-down", "kind:no-further-action", "kind:closed"]:
            with self.subTest(content=content):
                self.assertFalse(self._would_dispatch(content=content, bot=True, allow_bots="all"))

    def test_bot_actionable_routes_once_under_all(self):
        for content in ["kind:request please build X", "kind:handoff to you",
                        "kind:requires-ack on the deploy"]:
            with self.subTest(content=content):
                self.assertTrue(self._would_dispatch(content=content, bot=True, allow_bots="all"))

    # ── allow_bots=mentions ──────────────────────────────────────────
    def test_bot_terminal_dropped_under_mentions_when_mentioned(self):
        # Terminal post that @mentions us is still suppressed.
        self.assertFalse(
            self._would_dispatch(content="<@99999> kind:ack", bot=True,
                                 allow_bots="mentions", mentioned=True)
        )

    def test_bot_actionable_routes_under_mentions_when_mentioned(self):
        self.assertTrue(
            self._would_dispatch(content="<@99999> kind:request do the thing", bot=True,
                                 allow_bots="mentions", mentioned=True)
        )

    def test_bot_dropped_under_mentions_without_mention(self):
        # No mention → dropped at the permit, regardless of kind.
        self.assertFalse(
            self._would_dispatch(content="kind:request", bot=True,
                                 allow_bots="mentions", mentioned=False)
        )

    # ── humans never suppressed ──────────────────────────────────────
    def test_human_terminal_chatter_still_routes(self):
        for content in ["kind:ack", "ack", "final", "status done", "kind:no-op"]:
            with self.subTest(content=content):
                self.assertTrue(self._would_dispatch(content=content, bot=False, allow_bots="all"))

    # ── allow_bots=none unchanged ────────────────────────────────────
    def test_allow_bots_none_unchanged(self):
        # All other bots dropped at the permit; terminal guard never reached.
        self.assertFalse(self._would_dispatch(content="kind:request", bot=True, allow_bots="none"))
        self.assertFalse(self._would_dispatch(content="kind:ack", bot=True, allow_bots="none"))

    # ── loop smoke ───────────────────────────────────────────────────
    def test_loop_smoke_zero_dispatch_on_terminal_sequence(self):
        sequence = ["kind:ack", "kind:final", "kind:no-op", "kind:status",
                    "kind:ack", "kind:closed", "kind:standing-down"]
        dispatched = [c for c in sequence
                      if self._would_dispatch(content=c, bot=True, allow_bots="all")]
        self.assertEqual(dispatched, [], "terminal bot chatter must not dispatch")


if __name__ == "__main__":
    unittest.main()
