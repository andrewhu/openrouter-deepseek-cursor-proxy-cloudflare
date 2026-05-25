from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.trace import sha256_text
from deepseek_cursor_proxy.tunnel import CloudflareTunnel

TEST_API_KEY = "sk-test"
TEST_API_KEY_HASH = sha256_text(TEST_API_KEY)
DEFAULT_TUNNEL_STDOUT = b"INF Connection proxy.example.com registered\n"


class HttpServerFixture:
    def __init__(self, server: ThreadingHTTPServer) -> None:
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def post_json(
    url: str,
    payload: dict,
    *,
    authorization: str = f"Bearer {TEST_API_KEY}",
    timeout: float = 10,
) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@dataclass
class CloudflaredStartMocks:
    process: MagicMock
    write_text: MagicMock


@contextmanager
def mock_cloudflared_start(
    stdout_line: bytes = DEFAULT_TUNNEL_STDOUT,
    *,
    select_ready: bool = True,
    probe_public_health: bool = False,
):
    """Patch cloudflared Popen/select/read/path writes for ``CloudflareTunnel.start()``."""
    process = MagicMock()
    process.poll.return_value = None
    process.stdout.fileno.return_value = 99
    write_text = MagicMock()
    sel_return = ([99], [], []) if select_ready else ([], [], [])
    with (
        patch("deepseek_cursor_proxy.tunnel.shutil.which", return_value="/x/cloudflared"),
        patch("deepseek_cursor_proxy.tunnel.subprocess.Popen", return_value=process),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.write_text", write_text),
        patch("pathlib.Path.chmod"),
        patch("deepseek_cursor_proxy.tunnel.os.set_blocking"),
        patch("deepseek_cursor_proxy.tunnel.select.select", return_value=sel_return),
        patch(
            "deepseek_cursor_proxy.tunnel.os.read",
            return_value=stdout_line if select_ready else b"still starting\n",
        ),
    ):
        if probe_public_health:
            yield CloudflaredStartMocks(process=process, write_text=write_text)
        else:
            with patch.object(CloudflareTunnel, "_probe_public_health"):
                yield CloudflaredStartMocks(process=process, write_text=write_text)


def mock_tunnel_list_json(tunnels: list[dict]) -> str:
    return json.dumps(tunnels)
