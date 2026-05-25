"""Trace writer tests, both as a unit (writes/redacts files) and integrated
through the proxy (captures real request flow on disk)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer
from deepseek_cursor_proxy.trace import TraceWriter
from tests.support.fixtures import HttpServerFixture, TEST_API_KEY, TEST_API_KEY_HASH


class TraceWriterUnitTests(unittest.TestCase):
    def test_writes_manifest_and_numbered_request_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            writer = TraceWriter(temp_dir)
            first = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={"User-Agent": "Cursor/1.0"},
            )
            second = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={"User-Agent": "Cursor/1.0"},
            )
            first.finish("completed", http_status=200)
            second.finish("completed", http_status=200)

            self.assertTrue((writer.session_dir / "manifest.json").exists())
            self.assertTrue((writer.session_dir / "request-000001.json").exists())
            self.assertTrue((writer.session_dir / "request-000002.json").exists())
            self.assertEqual(
                stat.S_IMODE((writer.session_dir / "request-000001.json").stat().st_mode),
                0o600,
            )

    def test_sanitize_trace_omits_response_and_stream_bodies(self) -> None:
        with TemporaryDirectory() as temp_dir:
            writer = TraceWriter(temp_dir, sanitize_trace=True)
            trace = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={},
            )
            trace.record_upstream_response(
                status=200,
                body=b'{"choices":[]}',
            )
            trace.record_cursor_response(status=200, body=b'{"ok":true}')
            trace.record_stream_chunk(b"data: {}\n", b"data: {}\n")
            trace.finish("completed", http_status=200)
            payload = json.loads(trace.path.read_text(encoding="utf-8"))
            self.assertNotIn("choices", json.dumps(payload))
            self.assertEqual(payload["upstream"]["response"]["body_bytes"], len(b'{"choices":[]}'))
            self.assertIn("upstream_bytes", payload["upstream"]["stream"]["chunks"][0])

    def test_sanitized_response_hashes_use_raw_bytes(self) -> None:
        body = b"\xff"
        with TemporaryDirectory() as temp_dir:
            writer = TraceWriter(temp_dir, sanitize_trace=True)
            trace = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={},
            )
            trace.record_upstream_response(status=200, body=body)
            trace.record_cursor_response(status=200, body=body)
            trace.finish("completed", http_status=200)
            payload = json.loads(trace.path.read_text(encoding="utf-8"))

        expected = hashlib.sha256(body).hexdigest()
        self.assertEqual(payload["upstream"]["response"]["body_sha256"], expected)
        self.assertEqual(payload["cursor_response"]["body_sha256"], expected)

    def test_authorization_header_is_redacted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            writer = TraceWriter(temp_dir)
            trace = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={"Authorization": "Bearer sk-secret"},
            )
            trace.finish("completed", http_status=200)
            serialized = trace.path.read_text(encoding="utf-8")
            self.assertNotIn("sk-secret", serialized)
            payload = json.loads(serialized)
            self.assertEqual(payload["request"]["headers"]["Authorization"]["present"], True)
            self.assertIn("sha256", payload["request"]["headers"]["Authorization"])


class RecordCursorBodyBytesTests(unittest.TestCase):
    def _trace_payload(self, body: bytes, *, sanitize_trace: bool) -> dict[str, object]:
        with TemporaryDirectory() as temp_dir:
            writer = TraceWriter(temp_dir, sanitize_trace=sanitize_trace)
            trace = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={},
            )
            trace.record_cursor_body_bytes(body)
            trace.finish("completed", http_status=200)
            return json.loads(trace.path.read_text(encoding="utf-8"))

    def test_sanitize_trace_omits_body_for_invalid_json(self) -> None:
        payload = self._trace_payload(b"not json", sanitize_trace=True)
        self.assertEqual(payload["request"]["body_bytes"], len(b"not json"))
        self.assertNotIn("body", payload["request"])
        self.assertEqual(payload["request"]["body_omitted"]["reason"], "non_json")
        self.assertIn("body_sha256", payload["request"])

    def test_sanitize_trace_omits_body_for_non_dict_json(self) -> None:
        payload = self._trace_payload(b"[1,2,3]", sanitize_trace=True)
        self.assertEqual(payload["request"]["body_bytes"], len(b"[1,2,3]"))
        self.assertNotIn("body", payload["request"])
        self.assertEqual(payload["request"]["body_omitted"]["reason"], "non_object")
        self.assertIn("body_sha256", payload["request"])

    def test_unsanitized_trace_records_body_for_invalid_json(self) -> None:
        payload = self._trace_payload(b"not json", sanitize_trace=False)
        self.assertEqual(payload["request"]["body"], {"text": "not json"})

    def test_unsanitized_trace_records_body_for_non_dict_json(self) -> None:
        payload = self._trace_payload(b"[1,2,3]", sanitize_trace=False)
        self.assertEqual(payload["request"]["body"], {"text": "[1,2,3]"})

    def test_sanitize_trace_keeps_summary_for_valid_dict_json(self) -> None:
        secret = "super-secret-prompt-text"
        body = json.dumps(
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": secret}],
            }
        ).encode("utf-8")
        payload = self._trace_payload(body, sanitize_trace=True)
        serialized = json.dumps(payload)
        self.assertEqual(payload["request"]["body_bytes"], len(body))
        self.assertNotIn("body", payload["request"])
        self.assertIn("body_sha256", payload["request"])
        self.assertNotIn(secret, serialized)
        self.assertEqual(payload["request"]["summary"]["model"], "deepseek-v4-pro")
        self.assertEqual(
            payload["request"]["summary"]["messages"][0]["content"]["length"],
            len(secret),
        )


# ---------------------------------------------------------------------------
# Integration: trace writer attached to a running proxy.
# ---------------------------------------------------------------------------


class _CannedUpstream(BaseHTTPRequestHandler):
    """Returns a tool-call response for the first POST and a streamed
    reasoning response for the second."""

    requests: list[dict[str, object]] = []

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"id":"s","object":"chat.completion.chunk","choices":'
                b'[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"},'
                b'"finish_reason":null}]}\n\n'
            )
            self.wfile.write(
                b'data: {"id":"s","object":"chat.completion.chunk","choices":'
                b'[{"index":0,"delta":{"content":"answer"},"finish_reason":null}],'
                b'"usage":{"completion_tokens_details":{"reasoning_tokens":1}}}\n\n'
            )
            self.wfile.write(
                b'data: {"id":"s","object":"chat.completion.chunk",'
                b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        body = json.dumps(
            {
                "id": "tool",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "I need the date.",
                            "tool_calls": [
                                {
                                    "id": "call_date",
                                    "type": "function",
                                    "function": {
                                        "name": "get_date",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _read_single_trace(session_dir: Path) -> dict:
    deadline = time.monotonic() + 2
    files = sorted(session_dir.glob("request-*.json"))
    while not files and time.monotonic() < deadline:
        time.sleep(0.01)
        files = sorted(session_dir.glob("request-*.json"))
    if len(files) != 1:
        raise AssertionError(f"expected one trace, found {files}")
    return json.loads(files[0].read_text(encoding="utf-8"))


class TraceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        _CannedUpstream.requests = []
        self.upstream = HttpServerFixture(ThreadingHTTPServer(("127.0.0.1", 0), _CannedUpstream))
        self.store = ReasoningStore(":memory:")
        self.temp_dir = TemporaryDirectory()
        self.writer = TraceWriter(self.temp_dir.name, sanitize_trace=True)
        proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        proxy.config = ProxyConfig(
            upstream_base_url=self.upstream.url,
            proxy_api_key_hash=TEST_API_KEY_HASH,
        )
        proxy.reasoning_store = self.store
        proxy.trace_writer = self.writer
        self.proxy = HttpServerFixture(proxy)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()
        self.temp_dir.cleanup()

    def _post(self, payload: dict) -> dict:
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {TEST_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def test_rejected_invalid_bearer_trace_omits_prompt_text(self) -> None:
        secret = "do-not-persist-this-prompt"
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": secret}],
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer wrong-key",
                "Content-Type": "application/json",
            },
        )
        with self.assertRaises(HTTPError) as captured:
            urlopen(request, timeout=5)
        self.assertEqual(captured.exception.code, 401)
        captured.exception.read()

        trace = _read_single_trace(self.writer.session_dir)
        serialized = json.dumps(trace)
        self.assertEqual(trace["completion"]["status"], "rejected")
        self.assertEqual(trace["completion"]["http_status"], 401)
        self.assertNotIn("body", trace["request"])
        self.assertNotIn(secret, serialized)
        self.assertEqual(
            trace["request"]["summary"]["messages"][0]["content"]["length"],
            len(secret),
        )

    def test_traces_unsupported_post_path_with_body(self) -> None:
        request = Request(
            f"{self.proxy.url}/v1/summarize",
            data=json.dumps(
                {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "summarize"}],
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {TEST_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with self.assertRaises(HTTPError) as captured:
            urlopen(request, timeout=5)
        self.assertEqual(captured.exception.code, 404)
        captured.exception.read()

        trace = _read_single_trace(self.writer.session_dir)
        self.assertEqual(trace["request"]["method"], "POST")
        self.assertEqual(trace["request"]["path"], "/v1/summarize")
        self.assertNotIn("body", trace["request"])
        self.assertEqual(trace["request"]["summary"]["model"], "gpt-4o-mini")
        self.assertEqual(trace["completion"]["status"], "rejected")
        self.assertEqual(trace["completion"]["http_status"], 404)
        self.assertEqual(trace["transform"], {})
        self.assertEqual(_CannedUpstream.requests, [])

    def test_captures_non_streaming_replay_without_api_key(self) -> None:
        self._post(
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "What is tomorrow's date?"}],
            }
        )
        trace = _read_single_trace(self.writer.session_dir)
        serialized = json.dumps(trace)
        self.assertEqual(trace["completion"]["status"], "completed")
        self.assertNotIn("body", trace["request"])
        self.assertEqual(trace["request"]["summary"]["messages"][0]["content"]["length"], 24)
        self.assertIn("body_sha256", trace["upstream"]["response"])
        self.assertNotIn("choices", serialized)
        self.assertNotIn(TEST_API_KEY, serialized)

    def test_captures_streaming_replay_chunks(self) -> None:
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
        with urlopen(request, timeout=2) as response:
            response.read()
        trace = _read_single_trace(self.writer.session_dir)
        self.assertEqual(trace["completion"]["status"], "completed")
        self.assertGreater(trace["upstream"]["stream"]["chunks"][0]["upstream_bytes"], 0)
        self.assertGreater(trace["upstream"]["stream"]["chunks"][0]["cursor_bytes"], 0)

    def test_captures_recovery_diagnostics(self) -> None:
        """A request that triggers cold-cache recovery records the recovery
        steps + diagnostic counters in the trace."""
        self._post(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "old"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_x", "content": "result"},
                    {"role": "user", "content": "new"},
                ],
            }
        )
        trace = _read_single_trace(self.writer.session_dir)
        self.assertEqual(trace["transform"]["recovery_steps"][0]["strategy"], "latest_user")
        self.assertGreaterEqual(
            len([item for item in trace["transform"]["reasoning_diagnostics"] if item["missing"]]),
            1,
        )


if __name__ == "__main__":
    unittest.main()
