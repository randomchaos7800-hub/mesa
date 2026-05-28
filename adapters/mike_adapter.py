"""MikeAdapter — benchmarks Mike's full relay + memory pipeline via MESA.

Pipeline:
  reset()         → clears mesa_test user messages from Mike's session store
  inject(turns)   → saves turns into Mike's session store as mesa_test user
  ask(question)   → calls relay.respond() — full pipeline: context + tools + model

Supports write-side v2 observability via get_writes().
Retrieval-side trace is intentionally not exposed because Mike may use relay
tools and persistent state that are not faithfully observable from this adapter.

Scope: defaults to "full_production" (relay tools + prior memory allowed).
Pass scope="pure_injection" only if you have a fully isolated Mike instance.

Requires cha0tiktower reachable at http://100.120.50.35:8010/v1.
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
from mesa.core.types import MemoryWrite

logger = logging.getLogger(__name__)

SESSIONS_DB = Path.home() / "mike-memory" / "sessions.db"
TEST_USER_ID = "mesa_test"


class MikeAdapter(MemoryAdapter):
    """Benchmarks Mike's full relay pipeline.

    inject() feeds turns into Mike's session store; ask() calls relay.respond()
    which loads that context and may fire tool calls (recall_memory,
    search_lighthouse, etc.). This is Mike as he runs in production.
    """

    def __init__(self, scope: str = "full_production"):
        from relay.relay import get_relay
        from relay.sessions import SessionStore
        self._relay = get_relay()
        self._store = SessionStore(SESSIONS_DB)
        self._injected: list[dict] = []
        self.scope = scope  # "pure_injection" | "full_production"

    def get_scope(self) -> str:
        return self.scope

    def reset(self) -> None:
        self._injected = []
        try:
            with sqlite3.connect(str(SESSIONS_DB)) as conn:
                conn.execute("DELETE FROM messages WHERE user_id = ?", (TEST_USER_ID,))
        except Exception as e:
            logger.warning(f"reset: DB clear failed: {e}")

    def _record_turns(self, turns: list[dict], session_date: Optional[str] = None) -> None:
        for turn in turns:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if content:
                self._store.save_message(TEST_USER_ID, role, content)
                self._injected.append({"role": role, "content": content, "session_date": session_date})

    def inject(self, turns: list[dict]) -> None:
        self._record_turns(turns, session_date=None)

    def inject_session(self, turns: list[dict], session_date: Optional[str] = None) -> None:
        if session_date:
            logger.debug(f"inject_session: date={session_date} ({len(turns)} turns)")
        self._record_turns(turns, session_date=session_date)

    def ask(self, question: str) -> str:
        try:
            return self._relay.respond(question, TEST_USER_ID, interface="mesa")
        except Exception as e:
            logger.warning(f"ask failed: {e}")
            return "I don't have that information."

    def get_writes(self) -> list[MemoryWrite] | None:
        """Expose injected sessions as the observable writes for v2 runs.
        Note: Mike's real extraction + any tool-side memory writes are not
        captured here; this at least gives the injection boundary.
        """
        if not self._injected:
            return []
        writes = []
        for i, turn in enumerate(self._injected):
            date = turn.get("session_date")
            meta = {"role": turn["role"], "session_date": date} if date else {"role": turn["role"]}
            writes.append(MemoryWrite(
                memory_id=f"mesa_inject_{i}",
                text=f"{turn['role'].upper()}: {turn['content']}",
                metadata=meta
            ))
        return writes

    def ask_with_trace(self, question: str):
        """Deprecated for Mike.

        Mike's answer path may rely on relay tools, persistent memory, or other
        production state that this adapter cannot faithfully enumerate as a
        retrieval trace. Returning a synthetic trace would make official v2 runs
        look observable when they are not.
        """
        return None

    def stored_facts(self) -> list[str] | None:
        return [f"{t['role'].upper()}: {t['content'][:120]}" for t in self._injected]
