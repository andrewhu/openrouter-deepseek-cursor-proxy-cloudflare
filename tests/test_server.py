"""Server boundary, CLI, and operational tests.

Pure helper tests (gzip, summarize) and stub-handler tests (client
disconnect) live near the top. The bottom of the file boots a real proxy +
tiny upstream to exercise things that need the HTTP layer: bearer token
forwarding, oversized body, missing-bearer rejection, logging modes, and
streaming connection close.
"""

from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import gzip
import json
import logging
from pathlib import Path
import re
import time
from types import SimpleNamespace
import unittest
import zlib
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tests.support.fixtures import (
    HttpServerFixture,
    TEST_API_KEY,
    TEST_API_KEY_HASH,
    post_json,
)

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.logging import (
    ConsoleLogFormatter,
    TerminalSpinner,
)
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import (
    DeepSeekProxyHandler,
    DeepSeekProxyServer,
    _sanitize_for_logging,
    build_arg_parser,
    read_response_body,
    summarize_chat_payload,
)


# ---------------------------------------------------------------------------
# Stubs for fast in-process tests of internal handler methods
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes, encoding: str = "", status: int = 200) -> None:
        self._body = BytesIO(body)
        self.headers = {"Content-Encoding": encoding} if encoding else {}
        self.status = status

    def read(self) -> bytes:
        return self._body.read()


class _FakeStreamingResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self.readline_calls = 0

    def readline(self) -> bytes:
        self.readline_calls += 1
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FailingStreamingResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def readline(self) -> bytes:
        raise OSError("record layer failure")


class _BrokenPipeWfile:
    def write(self, body: bytes) -> None:
        raise BrokenPipeError("test disconnect")

    def flush(self) -> None:
        raise BrokenPipeError("test disconnect")


class _FakeConsole:
    def __init__(self, *, tty: bool) -> None:
        self.tty = tty
        self.writes: list[str] = []

    def isatty(self) -> bool:
        return self.tty

    def write(self, text: str) -> None:
        self.writes.append(text)

    def flush(self) -> None:
        return


def _make_handler_stub(wfile: object, **config: object) -> DeepSeekProxyHandler:
    handler = object.__new__(DeepSeekProxyHandler)
    handler.server = SimpleNamespace(
        config=ProxyConfig(**config),
        reasoning_store=ReasoningStore(":memory:"),
    )
    handler.wfile = wfile
    handler.close_connection = False
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None
    return handler


class SanitizeForLoggingTests(unittest.TestCase):
    def test_truncates_long_reasoning_content(self) -> None:
        long_reasoning = "R" * 500
        sanitized = _sanitize_for_logging({"messages": [{"role": "assistant", "reasoning_content": long_reasoning}]})
        value = sanitized["messages"][0]["reasoning_content"]
        self.assertTrue(value.endswith("..."))
        self.assertLess(len(value), len(long_reasoning))

    def test_truncates_long_tool_arguments(self) -> None:
        long_args = "A" * 500
        sanitized = _sanitize_for_logging(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "foo",
                                    "arguments": long_args,
                                }
                            }
                        ],
                    }
                ]
            }
        )
        value = sanitized["messages"][0]["tool_calls"][0]["function"]["arguments"]
        self.assertTrue(value.endswith("..."))
        self.assertLess(len(value), len(long_args))


# ---------------------------------------------------------------------------
# CLI / pure helpers
# ---------------------------------------------------------------------------


class CliAndHelperTests(unittest.TestCase):
    def test_cli_boolean_flags_have_on_and_off_forms(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--local",
                "--no-verbose",
                "--trace-dir",
                "/tmp/dcp-traces",
            ]
        )
        self.assertTrue(args.local)
        self.assertFalse(args.verbose)
        self.assertEqual(args.trace_dir, Path("/tmp/dcp-traces"))

    def test_cli_accepts_tunnel_url(self) -> None:
        args = build_arg_parser().parse_args(["--tunnel-url", "https://proxy.example.com"])
        self.assertEqual(args.tunnel_url, "https://proxy.example.com")

    def test_default_console_logging_hides_info_prefix_and_timestamp(self) -> None:
        formatter = ConsoleLogFormatter(verbose=False)
        info_record = logging.LogRecord(
            "deepseek_cursor_proxy",
            logging.INFO,
            __file__,
            1,
            "listening on %s",
            ("http://127.0.0.1:9000/v1",),
            None,
        )
        warning_record = logging.LogRecord(
            "deepseek_cursor_proxy",
            logging.WARNING,
            __file__,
            1,
            "trace logging enabled",
            (),
            None,
        )

        self.assertEqual(
            formatter.format(info_record),
            "listening on http://127.0.0.1:9000/v1",
        )
        self.assertEqual(formatter.format(warning_record), "WARNING trace logging enabled")

    def test_verbose_console_logging_shows_timestamp_and_level(self) -> None:
        formatter = ConsoleLogFormatter(verbose=True)
        record = logging.LogRecord(
            "deepseek_cursor_proxy",
            logging.INFO,
            __file__,
            1,
            "listening on %s",
            ("http://127.0.0.1:9000/v1",),
            None,
        )

        self.assertRegex(
            formatter.format(record),
            re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO listening on "),
        )

    def test_terminal_spinner_animates_only_for_tty(self) -> None:
        tty = _FakeConsole(tty=True)
        spinner = TerminalSpinner(enabled=True, text="└ {frame}", stream=tty, interval=0.001).start()
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline and not tty.writes:
            time.sleep(0.001)
        spinner.stop()

        output = "".join(tty.writes)
        self.assertIn(TerminalSpinner.hide_cursor, output)
        self.assertIn("└ ⠋", output)
        self.assertIn(TerminalSpinner.show_cursor, output)
        self.assertTrue(output.endswith(TerminalSpinner.show_cursor))

        non_tty = _FakeConsole(tty=False)
        TerminalSpinner(enabled=True, text="└ {frame}", stream=non_tty, interval=0.001).start().stop()
        self.assertEqual(non_tty.writes, [])

    def test_read_response_body_decodes_gzip_and_deflate(self) -> None:
        self.assertEqual(
            read_response_body(_FakeResponse(gzip.compress(b'{"ok":1}'), "gzip")),
            b'{"ok":1}',
        )
        self.assertEqual(
            read_response_body(_FakeResponse(zlib.compress(b'{"ok":1}'), "deflate")),
            b'{"ok":1}',
        )

    def test_summarize_chat_payload_omits_message_content(self) -> None:
        summary = summarize_chat_payload(
            {
                "model": "deepseek-v4-pro",
                "stream": True,
                "messages": [{"role": "user", "content": "secret prompt"}],
                "tools": [{"type": "function"}],
                "tool_choice": "auto",
            }
        )
        self.assertIn("model='deepseek-v4-pro'", summary)
        self.assertIn("messages=1", summary)
        self.assertNotIn("secret prompt", summary)


# ---------------------------------------------------------------------------
# Client-disconnect / upstream-failure stubs (no real HTTP needed)
# ---------------------------------------------------------------------------


class HandlerStubTests(unittest.TestCase):
    def test_regular_response_handles_client_disconnect(self) -> None:
        handler = _make_handler_stub(_BrokenPipeWfile())
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            }
        ).encode("utf-8")
        try:
            with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
                result = handler._proxy_regular_response(
                    _FakeResponse(body),
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "ns",
                )
        finally:
            handler.server.reasoning_store.close()
        self.assertFalse(result.sent)
        self.assertIn("sending upstream response body", "\n".join(captured.output))

    def test_streaming_response_stops_on_client_disconnect(self) -> None:
        handler = _make_handler_stub(_BrokenPipeWfile())
        chunk = {
            "id": "stream",
            "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
        }
        response = _FakeStreamingResponse(
            [
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )
        try:
            with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
                result = handler._proxy_streaming_response(
                    response,
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "ns",
                )
        finally:
            handler.server.reasoning_store.close()
        self.assertFalse(result.sent)
        self.assertEqual(response.readline_calls, 1)
        self.assertIn("sending streaming response chunk", "\n".join(captured.output))

    def test_streaming_response_handles_upstream_read_failure(self) -> None:
        handler = _make_handler_stub(BytesIO())
        try:
            with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
                result = handler._proxy_streaming_response(
                    _FailingStreamingResponse(),
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "ns",
                )
        finally:
            handler.server.reasoning_store.close()
        self.assertFalse(result.sent)
        self.assertIn("upstream streaming response read failed", "\n".join(captured.output))

    def test_streaming_always_mirrors_reasoning_into_collapsible_details(self) -> None:
        wfile = BytesIO()
        handler = _make_handler_stub(wfile)
        chunk = {
            "id": "stream",
            "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "delta": {"reasoning_content": "Need context."}}],
        }
        response = _FakeStreamingResponse(
            [
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )
        try:
            handler._proxy_streaming_response(
                response,
                "deepseek-v4-pro",
                [{"role": "user", "content": "hi"}],
                "ns",
            )
        finally:
            handler.server.reasoning_store.close()
        body = wfile.getvalue().decode("utf-8")
        self.assertIn("reasoning_content", body)
        self.assertIn("<details>", body)


# ---------------------------------------------------------------------------
# HTTP-level boundary tests: real proxy + tiny upstream
# ---------------------------------------------------------------------------


class _PlainFakeUpstream(BaseHTTPRequestHandler):
    """Returns a fixed plain response and records every request."""

    requests: list[dict[str, object]] = []
    auth_headers: list[str] = []
    delay_after_done: float = 0.0
    status_code: int = 200
    response: dict[str, object] = {}

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)
        self.__class__.auth_headers.append(self.headers.get("Authorization", ""))

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            if self.__class__.delay_after_done:
                time.sleep(self.__class__.delay_after_done)
            return

        body = json.dumps(self.__class__.response).encode("utf-8")
        self.send_response(self.__class__.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_BASE_RESPONSE: dict[str, object] = {
    "id": "x",
    "object": "chat.completion",
    "created": 1,
    "model": "deepseek-v4-pro",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_cache_hit_tokens": 12,
        "prompt_cache_miss_tokens": 8,
        "completion_tokens_details": {"reasoning_tokens": 3},
    },
}


class HttpBoundaryTests(unittest.TestCase):
    """Real-HTTP tests that don't fit the protocol suite: things the proxy
    must do at the HTTP boundary regardless of what DeepSeek answers."""

    def setUp(self) -> None:
        _PlainFakeUpstream.requests = []
        _PlainFakeUpstream.auth_headers = []
        _PlainFakeUpstream.delay_after_done = 0.0
        _PlainFakeUpstream.status_code = 200
        _PlainFakeUpstream.response = dict(_BASE_RESPONSE)
        self.upstream = HttpServerFixture(ThreadingHTTPServer(("127.0.0.1", 0), _PlainFakeUpstream))
        self.store = ReasoningStore(":memory:")
        proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        proxy.config = ProxyConfig(
            upstream_base_url=self.upstream.url,
            proxy_api_key_hash=TEST_API_KEY_HASH,
        )
        proxy.reasoning_store = self.store
        self.proxy = HttpServerFixture(proxy)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def _request(self) -> dict:
        return {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_rejects_missing_bearer_token(self) -> None:
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(self._request()).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 401)
        self.assertEqual(_PlainFakeUpstream.requests, [])

    def test_accepts_valid_api_key(self) -> None:
        status, _ = post_json(f"{self.proxy.url}/v1/chat/completions", self._request())
        self.assertEqual(status, 200)

    def test_rejects_invalid_api_key(self) -> None:
        status, payload = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            self._request(),
            authorization="Bearer sk-wrong",
        )
        self.assertEqual(status, 401)
        self.assertIn("Invalid", payload["error"]["message"])
        self.assertEqual(_PlainFakeUpstream.requests, [])

    def test_rejects_chat_when_tunnel_dead(self) -> None:
        self.proxy.server.tunnel = SimpleNamespace(
            dead=True,
            tunnel_url="https://proxy.example.com",
        )
        status, payload = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            self._request(),
            authorization=f"Bearer {TEST_API_KEY}",
        )
        self.assertEqual(status, 503)
        self.assertIn("tunnel", payload["error"]["message"].lower())
        self.assertEqual(_PlainFakeUpstream.requests, [])

    def test_rejects_models_when_tunnel_dead(self) -> None:
        self.proxy.server.tunnel = SimpleNamespace(
            dead=True,
            tunnel_url="https://proxy.example.com",
        )
        request = Request(
            f"{self.proxy.url}/v1/models",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=2)
        self.assertEqual(caught.exception.code, 503)
        body = json.loads(caught.exception.read().decode("utf-8"))
        self.assertIn("tunnel", body["error"]["message"].lower())

    def test_rejects_oversized_request_body(self) -> None:
        with patch("deepseek_cursor_proxy.server.DEFAULT_MAX_REQUEST_BODY_BYTES", 10):
            status, payload = post_json(f"{self.proxy.url}/v1/chat/completions", self._request())
        self.assertEqual(status, 413)
        self.assertIn("too large", payload["error"]["message"])
        self.assertEqual(_PlainFakeUpstream.requests, [])

    def test_forwards_openrouter_upstream_error_without_rewrite(self) -> None:
        _PlainFakeUpstream.status_code = 401
        _PlainFakeUpstream.response = {"error": {"message": "invalid OpenRouter API key", "code": 401}}
        status, payload = post_json(f"{self.proxy.url}/v1/chat/completions", self._request())
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["message"], "invalid OpenRouter API key")
        self.assertNotIn("choices", payload)

    def test_forwards_bearer_token_to_upstream(self) -> None:
        status, _ = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            self._request(),
            authorization=f"Bearer {TEST_API_KEY}",
        )
        self.assertEqual(status, 200)
        self.assertEqual(_PlainFakeUpstream.auth_headers[0], f"Bearer {TEST_API_KEY}")

    def test_streaming_response_closes_after_done_when_upstream_lingers(
        self,
    ) -> None:
        """Cursor relies on the proxy ending the SSE stream at [DONE], even
        if the upstream socket stays open."""
        _PlainFakeUpstream.delay_after_done = 2.0
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "stream": True,
                    "messages": [{"role": "user", "content": "stream"}],
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
        )
        started = time.monotonic()
        with urlopen(request, timeout=1) as response:
            body = response.read().decode("utf-8")
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIn("data: [DONE]", body)

    def test_normal_logging_summarizes_without_bodies_or_keys(self) -> None:
        with self.assertLogs("deepseek_cursor_proxy", level="INFO") as captured:
            status, _ = post_json(
                f"{self.proxy.url}/v1/chat/completions",
                self._request(),
                authorization=f"Bearer {TEST_API_KEY}",
            )
            # `└ stats` is emitted on the handler thread *after* the response
            # body hits the socket, so the client may return before it lands.
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not any("└ stats" in record for record in captured.output):
                time.sleep(0.01)
        output = "\n".join(captured.output)
        self.assertEqual(status, 200)
        self.assertIn("┌ request model=deepseek-v4-pro effort=xhigh messages=1", output)
        self.assertIn("├ context status=ok reasoning_context=0", output)
        self.assertIn("└ stats", output)
        self.assertNotIn(" tools=", output)
        self.assertNotIn("├ send", output)
        request_summary_line = output.split("┌ request", 1)[1].split("\n", 1)[0]
        self.assertNotIn('"content"', request_summary_line)
        self.assertNotIn("sk-from-cursor", output)

    def test_verbose_logging_includes_bodies_but_redacts_api_key(self) -> None:
        self.proxy.server.config = replace(self.proxy.server.config, verbose=True)
        with self.assertLogs("deepseek_cursor_proxy", level="INFO") as captured:
            post_json(
                f"{self.proxy.url}/v1/chat/completions",
                self._request(),
                authorization=f"Bearer {TEST_API_KEY}",
            )
        output = "\n".join(captured.output)
        self.assertIn("cursor request body", output)
        self.assertIn("upstream request body", output)
        self.assertNotIn("sk-from-cursor", output)

    def test_models_lists_only_deepseek_v4_pro(self) -> None:
        request = Request(
            f"{self.proxy.url}/v1/models",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = [item["id"] for item in payload["data"]]
        self.assertEqual(model_ids, ["deepseek-v4-pro"])

    def test_healthz_returns_ok_with_auth(self) -> None:
        request = Request(
            f"{self.proxy.url}/healthz",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        with urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["ok"], True)

    def test_healthz_requires_auth(self) -> None:
        with self.assertRaises(HTTPError) as caught:
            urlopen(f"{self.proxy.url}/healthz", timeout=2)
        self.assertEqual(caught.exception.code, 401)

    def test_main_exits_without_proxy_api_key_hash(self) -> None:
        from deepseek_cursor_proxy.server import main

        config = replace(
            ProxyConfig(),
            tunnel_url="https://proxy.example.com",
            proxy_api_key_hash=None,
        )
        with patch("deepseek_cursor_proxy.server.ProxyConfig.from_file", return_value=config):
            with patch("deepseek_cursor_proxy.server.configure_logging"):
                exit_code = main([])
        self.assertEqual(exit_code, 2)

    def test_main_local_mode_skips_tunnel_requirements(self) -> None:
        from deepseek_cursor_proxy.server import main

        config = replace(
            ProxyConfig(),
            proxy_api_key_hash=TEST_API_KEY_HASH,
            tunnel_url=None,
        )
        with patch("deepseek_cursor_proxy.server.ProxyConfig.from_file", return_value=config):
            with patch("deepseek_cursor_proxy.server.configure_logging"):
                with patch("deepseek_cursor_proxy.server.ReasoningStore") as store_cls:
                    store = MagicMock()
                    store_cls.return_value = store
                    with patch("deepseek_cursor_proxy.server.DeepSeekProxyServer") as server_cls:
                        server = MagicMock()
                        server_cls.return_value = server
                        with patch("deepseek_cursor_proxy.server.CloudflareTunnel") as tunnel_cls:
                            exit_code = main(["--local"])
        self.assertEqual(exit_code, 0)
        tunnel_cls.assert_not_called()
        server.serve_forever.assert_called_once()


if __name__ == "__main__":
    unittest.main()
