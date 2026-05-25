"""Helpers for opt-in OpenRouter smoke tests (real API key, billed usage)."""

from __future__ import annotations

import json
import os
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import DEFAULT_UPSTREAM_BASE_URL, ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer
from deepseek_cursor_proxy.trace import sha256_text

from tests.support.fixtures import post_json

LIVE_SKIP_REASON = (
    "set RUN_LIVE_OPENROUTER_TESTS=1 and LIVE_OPENROUTER_KEY " "(or OPENROUTER_API_KEY) to run smoke tests"
)

LIVE_REQUEST_TIMEOUT = 180.0


def live_openrouter_enabled() -> bool:
    return os.getenv("RUN_LIVE_OPENROUTER_TESTS") == "1" and bool(live_openrouter_key())


def live_openrouter_key() -> str | None:
    key = os.getenv("LIVE_OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
    if key is None:
        return None
    stripped = key.strip()
    return stripped if stripped else None


def live_proxy_config(api_key: str) -> ProxyConfig:
    return ProxyConfig(
        upstream_base_url=DEFAULT_UPSTREAM_BASE_URL,
        proxy_api_key_hash=sha256_text(api_key),
    )


def live_post_json(
    url: str,
    payload: dict,
    *,
    api_key: str,
    timeout: float = LIVE_REQUEST_TIMEOUT,
) -> tuple[int, dict]:
    return post_json(
        url,
        payload,
        authorization=f"Bearer {api_key}",
        timeout=timeout,
    )


def live_get(
    url: str,
    *,
    api_key: str | None = None,
    timeout: float = 10,
) -> tuple[int, dict]:
    headers: dict[str, str] = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, method="GET", headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body) if body else {}


def skip_unless_live_openrouter(test_class: type[unittest.TestCase]) -> type:
    return unittest.skipUnless(live_openrouter_enabled(), LIVE_SKIP_REASON)(test_class)


class LiveProxyFixture:
    """OpenRouter-only by design; see README "Hardcoded behavior"."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.store = ReasoningStore(":memory:")
        server = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        server.config = live_proxy_config(api_key)
        server.reasoning_store = self.store
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def start(self) -> LiveProxyFixture:
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.store.close()
