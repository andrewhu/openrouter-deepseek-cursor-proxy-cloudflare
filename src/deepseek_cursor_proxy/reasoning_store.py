from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


def normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        tool_call = {}
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}

    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    normalized: dict[str, Any] = {
        "id": str(tool_call.get("id") or ""),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": arguments,
        },
    }
    if not normalized["id"]:
        normalized.pop("id")
    return normalized


def tool_call_signature(tool_call: dict[str, Any]) -> str:
    normalized = normalize_tool_call(tool_call)
    normalized.pop("id", None)
    canonical = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict) and tool_call.get("id"):
            ids.append(str(tool_call["id"]))
    return ids


def tool_call_names(message: dict[str, Any]) -> list[str]:
    return [name for _, name in tool_call_name_entries(message)]


def tool_call_name_entries(message: dict[str, Any]) -> list[tuple[int, str]]:
    """Indexed tool names; disambiguates duplicate function names in one assistant turn."""
    entries: list[tuple[int, str]] = []
    for index, tool_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and function.get("name"):
            entries.append((index, str(function["name"])))
    return entries


def scoped_tool_name_key(scope: str, index: int, tool_name: str) -> str:
    return f"scope:{scope}:tool_name:{index}:{tool_name}"


def portable_tool_name_key(cache_namespace: str, turn_signature: str, index: int, tool_name: str) -> str:
    return f"namespace:{cache_namespace}:turn:{turn_signature}:" f"tool_name:{index}:{tool_name}"


def message_signature(message: dict[str, Any]) -> str:
    tool_calls = [
        normalize_tool_call(tool_call) for tool_call in (message.get("tool_calls") or []) if isinstance(tool_call, dict)
    ]
    payload = {
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_scope_message(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {"role": message.get("role")}
    for key in ("content", "name", "tool_call_id", "prefix"):
        if key in message:
            canonical[key] = message[key]
    if message.get("tool_calls"):
        canonical["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in message.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
    return canonical


def conversation_scope(messages: list[dict[str, Any]], namespace: str = "") -> str:
    scope_messages = [canonical_scope_message(message) for message in messages]
    return conversation_scope_from_canonical(scope_messages, namespace)


def conversation_scope_from_canonical(scope_messages: list[dict[str, Any]], namespace: str = "") -> str:
    payload: Any = scope_messages
    if namespace:
        payload = {"namespace": namespace, "messages": scope_messages}
    return _sha256_json(payload)


def turn_context_signature(prior_messages: list[dict[str, Any]]) -> str:
    last_user_index = next(
        (index for index in range(len(prior_messages) - 1, -1, -1) if prior_messages[index].get("role") == "user"),
        -1,
    )
    start_index = 0
    if last_user_index != -1:
        start_index = last_user_index
        while start_index > 0 and prior_messages[start_index - 1].get("role") == "user":
            start_index -= 1

    context_messages = [
        canonical_scope_message(message) for message in prior_messages[start_index:] if message.get("role") != "system"
    ]
    return _sha256_json(context_messages)


def scoped_reasoning_keys(message: dict[str, Any], scope: str) -> list[str]:
    keys = [f"scope:{scope}:signature:{message_signature(message)}"]
    keys.extend(f"scope:{scope}:tool_call:{tool_call_id}" for tool_call_id in tool_call_ids(message))
    keys.extend(
        f"scope:{scope}:tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    # Recovery-of-last-resort key. Catches the case where a streaming response
    # was interrupted (user pressed Stop) before the tool_call.id chunk arrived,
    # so neither tool_call_id nor tool_call_signature (which canonicalizes
    # arguments) survives the round-trip through Cursor's transcript.
    keys.extend(scoped_tool_name_key(scope, index, tool_name) for index, tool_name in tool_call_name_entries(message))
    return keys


def portable_reasoning_keys(
    message: dict[str, Any],
    cache_namespace: str,
    prior_messages: list[dict[str, Any]],
) -> list[str]:
    if not cache_namespace:
        return []

    turn_signature = turn_context_signature(prior_messages)
    keys = [f"namespace:{cache_namespace}:turn:{turn_signature}:" f"signature:{message_signature(message)}"]
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:" f"tool_call:{tool_call_id}"
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:" f"tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    keys.extend(
        portable_tool_name_key(cache_namespace, turn_signature, index, tool_name)
        for index, tool_name in tool_call_name_entries(message)
    )
    return keys


def reasoning_lookup_key_specs(
    message: dict[str, Any],
    scope: str,
    cache_namespace: str = "",
    prior_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "kind": "message_signature",
            "key": f"scope:{scope}:signature:{message_signature(message)}",
            "portable": False,
            "hit": False,
        }
    ]
    specs.extend(
        {
            "kind": "tool_call_id",
            "tool_call_id": tool_call_id,
            "key": f"scope:{scope}:tool_call:{tool_call_id}",
            "portable": False,
            "hit": False,
        }
        for tool_call_id in tool_call_ids(message)
    )
    specs.extend(
        {
            "kind": "tool_call_signature",
            "function_name": str((tool_call.get("function") or {}).get("name") or ""),
            "key": (f"scope:{scope}:tool_call_signature:" f"{tool_call_signature(tool_call)}"),
            "portable": False,
            "hit": False,
        }
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    specs.extend(
        {
            "kind": "tool_name",
            "tool_index": index,
            "function_name": tool_name,
            "key": scoped_tool_name_key(scope, index, tool_name),
            "portable": False,
            "hit": False,
        }
        for index, tool_name in tool_call_name_entries(message)
    )
    if cache_namespace and prior_messages is not None:
        turn_signature = turn_context_signature(prior_messages)
        specs.append(
            {
                "kind": "portable_message_signature",
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:" f"signature:{message_signature(message)}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
        )
        specs.extend(
            {
                "kind": "portable_tool_call_id",
                "tool_call_id": tool_call_id,
                "key": (f"namespace:{cache_namespace}:turn:{turn_signature}:" f"tool_call:{tool_call_id}"),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for tool_call_id in tool_call_ids(message)
        )
        specs.extend(
            {
                "kind": "portable_tool_call_signature",
                "function_name": str((tool_call.get("function") or {}).get("name") or ""),
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"tool_call_signature:{tool_call_signature(tool_call)}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for tool_call in (message.get("tool_calls") or [])
            if isinstance(tool_call, dict)
        )
        specs.extend(
            {
                "kind": "portable_tool_name",
                "tool_index": index,
                "function_name": tool_name,
                "key": portable_tool_name_key(cache_namespace, turn_signature, index, tool_name),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for index, tool_name in tool_call_name_entries(message)
        )
    return specs


class ReasoningStore:
    def __init__(
        self,
        reasoning_content_path: str | Path,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.max_age_seconds = max_age_seconds
        self.max_rows = max_rows
        if str(reasoning_content_path) == ":memory:":
            self.reasoning_content_path: str | Path = ":memory:"
        else:
            self.reasoning_content_path = Path(reasoning_content_path).expanduser()
            self.reasoning_content_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._last_prune_at = 0.0
        self._last_put_ts = 0.0
        self._conn = sqlite3.connect(self.reasoning_content_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA mmap_size=268435456")
        if isinstance(self.reasoning_content_path, Path):
            self.reasoning_content_path.chmod(0o600)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                key TEXT PRIMARY KEY,
                reasoning TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reasoning_cache_created_at " "ON reasoning_cache(created_at)"
        )
        self._conn.commit()
        self.prune()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except (sqlite3.Error, OSError):
                pass

    def put(self, key: str, reasoning: str, message: dict[str, Any]) -> None:
        if not isinstance(reasoning, str):
            return
        message_json = json.dumps(message, ensure_ascii=False, sort_keys=True)
        with self._lock:
            ts = time.time()
            if ts <= self._last_put_ts:
                ts = self._last_put_ts + 1e-6
            self._last_put_ts = ts
            self._put_locked(key, reasoning, message_json, ts)
            self._maybe_prune_locked(force=True)
            self._conn.commit()

    def _maybe_prune_locked(self, *, force: bool = False) -> None:
        now = time.time()
        if force or (now - self._last_prune_at) >= 1.0:
            self._prune_locked()
            self._last_prune_at = now

    def _put_locked(self, key: str, reasoning: str, message_json: str, ts: float) -> None:
        self._conn.execute(
            """
            INSERT INTO reasoning_cache(key, reasoning, message_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                reasoning = excluded.reasoning,
                message_json = excluded.message_json,
                created_at = excluded.created_at
            """,
            (key, reasoning, message_json, ts),
        )

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT reasoning FROM reasoning_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def store_assistant_message(
        self,
        message: dict[str, Any],
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        if message.get("role") != "assistant":
            return 0
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            return 0

        keys = scoped_reasoning_keys(message, scope)
        if prior_messages is not None:
            keys.extend(portable_reasoning_keys(message, cache_namespace, prior_messages))
        keys = list(dict.fromkeys(keys))
        message_json = json.dumps(message, ensure_ascii=False, sort_keys=True)
        ts = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for key in keys:
                    self._put_locked(key, reasoning, message_json, ts)
                self._maybe_prune_locked(force=False)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return len(keys)

    def lookup_for_message(
        self,
        message: dict[str, Any],
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> str | None:
        keys = scoped_reasoning_keys(message, scope)
        if prior_messages is not None:
            keys.extend(portable_reasoning_keys(message, cache_namespace, prior_messages))
        for key in keys:
            reasoning = self.get(key)
            if reasoning is not None:
                return reasoning
        return None

    def backfill_portable_aliases(
        self,
        message: dict[str, Any],
        reasoning: str,
        cache_namespace: str,
        prior_messages: list[dict[str, Any]],
    ) -> int:
        if not isinstance(reasoning, str):
            return 0
        keys = portable_reasoning_keys(message, cache_namespace, prior_messages)
        if not keys:
            return 0
        message_with_reasoning = dict(message)
        message_with_reasoning["reasoning_content"] = reasoning
        message_json = json.dumps(message_with_reasoning, ensure_ascii=False, sort_keys=True)
        unique_keys = list(dict.fromkeys(keys))
        ts = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for key in unique_keys:
                    self._put_locked(key, reasoning, message_json, ts)
                self._maybe_prune_locked(force=False)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return len(unique_keys)

    def clear(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM reasoning_cache").fetchone()
            count = int(row[0] if row else 0)
            self._conn.execute("DELETE FROM reasoning_cache")
            self._conn.commit()
        return count

    def prune(self) -> int:
        with self._lock:
            deleted = self._prune_locked()
            self._conn.commit()
        return deleted

    def _prune_locked(self) -> int:
        deleted = 0
        if self.max_age_seconds is not None and self.max_age_seconds > 0:
            cutoff = time.time() - self.max_age_seconds
            cursor = self._conn.execute(
                "DELETE FROM reasoning_cache WHERE created_at < ?",
                (cutoff,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        if self.max_rows is not None and self.max_rows > 0:
            cursor = self._conn.execute(
                """
                DELETE FROM reasoning_cache
                WHERE key NOT IN (
                    SELECT key
                    FROM reasoning_cache
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (self.max_rows,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0
        return deleted
