"""MikeAdapter — benchmarks Mike's full relay + memory pipeline via MESA.

Pipeline:
  reset()         → clears mesa_test user messages from Mike's session store
  inject(turns)   → saves turns into Mike's session store as mesa_test user
  ask(question)   → calls relay.respond() — full pipeline: context + tools + model

This tests Mike as actually deployed: injected sessions become context, and Mike
may also draw on his accumulated memory (LIGHTHOUSE, knowledge graph) via tool calls.
Requires cha0tiktower to be reachable at http://100.120.50.35:8081/v1.
"""

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Add Mike to Python path
MIKE_ROOT = Path.home() / "mike"
if str(MIKE_ROOT) not in sys.path:
    sys.path.insert(0, str(MIKE_ROOT))

# Load Mike's secrets so the relay can authenticate
from dotenv import load_dotenv
_secrets = MIKE_ROOT / "config" / "secrets.env"
if _secrets.exists():
    load_dotenv(_secrets, override=True)

from mesa.adapter import MemoryAdapter

logger = logging.getLogger(__name__)

SESSIONS_DB = Path.home() / "mike-memory" / "sessions.db"
TEST_USER_ID = "mesa_test"


class MikeAdapter(MemoryAdapter):
    """Benchmarks Mike's full relay pipeline.

    inject() feeds turns into Mike's session store; ask() calls relay.respond()
    which loads that context and may fire tool calls (recall_memory,
    search_lighthouse, etc.). This is Mike as he runs in production.
    """

    def __init__(self):
        from relay.relay import get_relay
        from relay.sessions import SessionStore
        self._relay = get_relay()
        self._store = SessionStore(SESSIONS_DB)
        self._injected: list[dict] = []

    def reset(self) -> None:
        self._injected = []
        try:
            with sqlite3.connect(str(SESSIONS_DB)) as conn:
                conn.execute("DELETE FROM messages WHERE user_id = ?", (TEST_USER_ID,))
        except Exception as e:
            logger.warning(f"reset: DB clear failed: {e}")

    def inject(self, turns: list[dict]) -> None:
        for turn in turns:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if content:
                self._store.save_message(TEST_USER_ID, role, content)
                self._injected.append({"role": role, "content": content})

    def inject_session(self, turns: list[dict], session_date: Optional[str] = None) -> None:
        if session_date:
            logger.debug(f"inject_session: date={session_date} ({len(turns)} turns)")
        self.inject(turns)

    def ask(self, question: str) -> str:
        try:
            return self._relay.respond(question, TEST_USER_ID, interface="mesa")
        except Exception as e:
            logger.warning(f"ask failed: {e}")
            return "I don't have that information."

    def stored_facts(self) -> list[str] | None:
        return [f"{t['role'].upper()}: {t['content'][:100]}" for t in self._injected]
