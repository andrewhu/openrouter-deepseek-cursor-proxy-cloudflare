from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import re
import select
import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse

import urllib3

from .logging import LOG


CONNECTION_REGISTERED_PATTERN = re.compile(r"Connection\s+.*\s+registered", re.IGNORECASE)
PUBLIC_HEALTH_PATH = "/v1/healthz"
HEALTH_PATHS = frozenset({"/healthz", PUBLIC_HEALTH_PATH})
PUBLIC_HEALTH_TIMEOUT = 10.0
MAX_RESTART_BACKOFF_SECONDS = 16.0
_PROBE_POOL = urllib3.PoolManager(maxsize=1)


@dataclass(frozen=True)
class NamedTunnelInfo:
    tunnel_id: str
    credentials_file: Path


def local_tunnel_target(host: str, port: int) -> str:
    local_host = host.strip() or "127.0.0.1"
    if local_host in {"0.0.0.0", "::"}:
        local_host = "127.0.0.1"
    if ":" in local_host and not local_host.startswith("["):
        local_host = f"[{local_host}]"
    return f"http://{local_host}:{port}"


@dataclass
class CloudflareTunnel:
    target_url: str
    tunnel_name: str
    tunnel_url: str
    command: str = "cloudflared"
    startup_timeout: float = 15.0
    max_restarts: int = 5
    probe_public_health: bool = True

    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    _tunnel_config_path: Path | None = field(default=None, init=False, repr=False)
    _tunnel_info: NamedTunnelInfo | None = field(default=None, init=False, repr=False)
    _monitor_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _monitor_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    dead: bool = field(default=False, init=False, repr=False)

    TUNNEL_CONFIG_YAML = """\
tunnel: {tunnel_id}
credentials-file: {credentials_file}

ingress:
  - hostname: {hostname}
    service: {target_url}
  - service: http_status:404
"""

    def __post_init__(self) -> None:
        self._validate_tunnel_url(self.tunnel_url)

    @staticmethod
    def _validate_tunnel_url(tunnel_url: str) -> None:
        parsed = urlparse(tunnel_url)
        if parsed.scheme != "https":
            raise ValueError(f"tunnel_url must use https:// scheme, got: {tunnel_url}")
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise ValueError(f"tunnel_url must include a hostname, got: {tunnel_url}")
        if "trycloudflare.com" in hostname:
            raise ValueError(
                "tunnel_url must be a custom hostname (e.g. https://deepseek.yourdomain.com), "
                "not a trycloudflare.com quick-tunnel URL. "
                "Quick Tunnels do not support Server-Sent Events (SSE) required by Cursor streaming. "
                "See: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/"
            )

    def start(self) -> str:
        if shutil.which(self.command) is None:
            raise RuntimeError(
                "cloudflared is not installed or is not on PATH. "
                "Install it with `brew install cloudflared`, then run "
                "`cloudflared tunnel login` once."
            )

        self.dead = False
        self._tunnel_info = lookup_named_tunnel(self.command, self.tunnel_name)
        self._run_cloudflared()
        self._start_monitor()
        return self.tunnel_url

    def _run_cloudflared(self) -> None:
        parsed = urlparse(self.tunnel_url)
        hostname = parsed.hostname or ""
        self._tunnel_config_path = self._write_tunnel_config(hostname)
        argv = [
            self.command,
            "tunnel",
            "--config",
            str(self._tunnel_config_path),
            "run",
            self.tunnel_name,
        ]
        self.process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            self._wait_for_connection()
            if self.probe_public_health:
                self._probe_public_health()
        except BaseException:
            if self.process is not None:
                _terminate_process(self.process)
            self.process = None
            raise

    def _write_tunnel_config(self, hostname: str) -> Path:
        if self._tunnel_info is None:
            raise RuntimeError("tunnel info not resolved; call _resolve_named_tunnel first")
        tunnel_dir = _tunnel_config_dir()
        tunnel_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        config_path = tunnel_dir / f"{self.tunnel_name}.yml"

        content = self.TUNNEL_CONFIG_YAML.format(
            tunnel_id=self._tunnel_info.tunnel_id,
            credentials_file=str(self._tunnel_info.credentials_file),
            hostname=hostname,
            target_url=self.target_url,
        )
        config_path.write_text(content)
        config_path.chmod(0o600)
        LOG.info(
            "wrote tunnel config to %s (hostname=%s, target=%s, credentials=%s)",
            config_path,
            hostname,
            self.target_url,
            self._tunnel_info.credentials_file,
        )
        return config_path

    def _wait_for_connection(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        fd = self.process.stdout.fileno()
        os.set_blocking(fd, False)
        try:
            buf = b""
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    try:
                        buf += os.read(fd, 65536)
                    except BlockingIOError:
                        pass
                    stdout = buf.decode("utf-8", errors="replace")
                    raise RuntimeError(f"cloudflared exited before connecting:\n{stdout}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                readable, _, _ = select.select([fd], [], [], min(0.5, remaining))
                if fd not in readable:
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    time.sleep(0.1)
                    continue
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").rstrip()
                    LOG.info("cloudflared: %s", line)
                    if CONNECTION_REGISTERED_PATTERN.search(line):
                        return
        finally:
            try:
                os.set_blocking(fd, True)
            except OSError:
                pass
        raise RuntimeError("Timed out waiting for cloudflared tunnel connection " f"(tunnel_name={self.tunnel_name})")

    def _probe_public_health(self) -> None:
        health_url = f"{self.tunnel_url.rstrip('/')}{PUBLIC_HEALTH_PATH}"
        try:
            response = _PROBE_POOL.request(
                "GET",
                health_url,
                timeout=urllib3.Timeout(total=PUBLIC_HEALTH_TIMEOUT),
            )
        except urllib3.exceptions.HTTPError as exc:
            raise RuntimeError(f"public tunnel health probe failed for {health_url}: {exc}") from exc
        if response.status != 200:
            raise RuntimeError(f"public tunnel health probe returned HTTP {response.status} " f"for {health_url}")
        LOG.info("public tunnel health probe OK: %s", health_url)

    def stop(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        if self.process is None or self.process.poll() is not None:
            self._cleanup_config()
            return
        LOG.info("stopping cloudflared tunnel")
        _terminate_process(self.process)
        self.process = None
        self._cleanup_config()

    def _start_monitor(self) -> None:
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name=f"tunnel-monitor-{self.tunnel_name}",
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        restarts = 0
        while not self._monitor_stop.wait(timeout=2):
            if self.process is None:
                continue
            rc = self.process.poll()
            if rc is None:
                continue
            if self._monitor_stop.is_set():
                return
            restarts += 1
            if restarts > self.max_restarts:
                self.dead = True
                LOG.warning(
                    "cloudflared tunnel %s is DEAD after %d restarts "
                    "(max_restarts=%d). Chat requests will return HTTP 503 until "
                    "you restart deepseek-cursor-proxy. tunnel_url=%s",
                    self.tunnel_name,
                    restarts,
                    self.max_restarts,
                    self.tunnel_url,
                )
                return
            LOG.warning(
                "cloudflared tunnel %s exited with code %d; restarting in %.0fs (attempt %d/%d)",
                self.tunnel_name,
                rc,
                self._restart_backoff_seconds(restarts),
                restarts,
                self.max_restarts,
            )
            if self._monitor_stop.wait(timeout=self._restart_backoff_seconds(restarts)):
                return
            try:
                self._restart_tunnel()
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                self.dead = True
                LOG.error(
                    "failed to restart tunnel %s: %s. Tunnel is DEAD.",
                    self.tunnel_name,
                    exc,
                )
                return

    def _restart_backoff_seconds(self, restart_attempt: int) -> float:
        return min(2 ** max(restart_attempt, 1), MAX_RESTART_BACKOFF_SECONDS)

    def _restart_tunnel(self) -> None:
        if self.process is not None:
            _terminate_process(self.process)
            self.process = None
        self._run_cloudflared()

    def _cleanup_config(self) -> None:
        if self._tunnel_config_path is not None:
            try:
                self._tunnel_config_path.unlink(missing_ok=True)
            except OSError:
                pass


def lookup_named_tunnel(command: str, tunnel_name: str) -> NamedTunnelInfo:
    """Resolve tunnel UUID and per-tunnel credentials JSON from cloudflared."""
    result = subprocess.run(
        [command, "tunnel", "list", "--output", "json"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list Cloudflare tunnels: {result.stderr or result.stdout}")
    tunnels = _parse_tunnel_list_json(result.stdout)
    if tunnels is None:
        tunnels = _parse_tunnel_list_text(result.stdout)
    for entry in tunnels:
        if entry.get("name") == tunnel_name:
            tunnel_id = str(entry.get("id") or "").strip()
            if not tunnel_id:
                raise RuntimeError(f"Named tunnel '{tunnel_name}' has no id in cloudflared output")
            credentials_file = _credentials_path_for_tunnel_id(tunnel_id)
            if not credentials_file.is_file():
                raise RuntimeError(
                    f"Tunnel credentials not found at {credentials_file}. "
                    f"Create the tunnel with: cloudflared tunnel create {tunnel_name}"
                )
            return NamedTunnelInfo(
                tunnel_id=tunnel_id,
                credentials_file=credentials_file,
            )
    raise RuntimeError(
        f"Named tunnel '{tunnel_name}' not found. " f"Create it with: cloudflared tunnel create {tunnel_name}"
    )


def _parse_tunnel_list_json(stdout: str) -> list[dict[str, str]] | None:
    try:
        loaded = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, list):
        entries = loaded
    elif isinstance(loaded, dict):
        entries = loaded.get("tunnels") or loaded.get("result") or []
    else:
        return None
    if not isinstance(entries, list):
        return None
    normalized: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "id": str(entry.get("id") or ""),
                "name": str(entry.get("name") or ""),
            }
        )
    return normalized


def _parse_tunnel_list_text(stdout: str) -> list[dict[str, str]]:
    """Fallback when --output json is unavailable."""
    entries: list[dict[str, str]] = []
    uuid_re = re.compile(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        re.IGNORECASE,
    )
    for line in stdout.splitlines():
        match = uuid_re.search(line)
        if not match:
            continue
        tunnel_id = match.group(1)
        name = line.replace(tunnel_id, "").strip()
        if name:
            entries.append({"id": tunnel_id, "name": name})
    return entries


def _credentials_path_for_tunnel_id(tunnel_id: str) -> Path:
    return Path.home() / ".cloudflared" / f"{tunnel_id}.json"


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _tunnel_config_dir() -> Path:
    return Path.home() / ".deepseek-cursor-proxy" / "tunnels"
