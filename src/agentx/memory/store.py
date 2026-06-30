"""Two-tier agent memory — dependency-free, works with any framework.

* ``ConversationMemory`` — short-term, in-process windowed buffer of turns.
* ``LongTermMemory``    — append-only JSONL persisted per session, survives restarts.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationMemory:
    """A rolling window of the most recent (role, content) turns."""

    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        self._turns.append((role, content))

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_ai(self, content: str) -> None:
        self.add("assistant", content)

    def as_messages(self) -> list[dict]:
        """Return turns as chat-style message dicts."""
        return [{"role": r, "content": c} for r, c in self._turns]

    def transcript(self) -> str:
        return "\n".join(f"{r}: {c}" for r, c in self._turns)

    def clear(self) -> None:
        self._turns.clear()


class LongTermMemory:
    """Append-only JSONL memory keyed by session id."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, role: str, content: str, **meta) -> dict:
        event = {"ts": _now(), "role": role, "content": content, **meta}
        with _lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        return event

    def history(self, limit: int | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-limit:] if limit else rows

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
