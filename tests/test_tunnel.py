from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepseek_cursor_proxy.tunnel import (
    CloudflareTunnel,
    NamedTunnelInfo,
    local_tunnel_target,
    lookup_named_tunnel,
    _credentials_path_for_tunnel_id,
    _parse_tunnel_list_json,
)
from tests.support.fixtures import mock_cloudflared_start, mock_tunnel_list_json


TUNNEL_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
TUNNEL_INFO = NamedTunnelInfo(
    tunnel_id=TUNNEL_UUID,
    credentials_file=_credentials_path_for_tunnel_id(TUNNEL_UUID),
)


def _patch_tunnel_lookup(test_case: unittest.TestCase) -> None:
    test_case.lookup_patcher = patch(
        "deepseek_cursor_proxy.tunnel.lookup_named_tunnel",
        return_value=TUNNEL_INFO,
    )
    test_case.lookup_patcher.start()
    test_case.addCleanup(test_case.lookup_patcher.stop)


class TunnelTests(unittest.TestCase):
    def test_local_tunnel_target_uses_loopback_for_wildcard_hosts(self) -> None:
        self.assertEqual(local_tunnel_target("0.0.0.0", 9000), "http://127.0.0.1:9000")
        self.assertEqual(local_tunnel_target("::", 9000), "http://127.0.0.1:9000")

    def test_local_tunnel_target_formats_ipv6_hosts(self) -> None:
        self.assertEqual(local_tunnel_target("::1", 9000), "http://[::1]:9000")

    def test_tunnel_url_must_be_https(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            CloudflareTunnel(
                "http://127.0.0.1:9000",
                tunnel_name="t",
                tunnel_url="http://proxy.example.com",
            )
        self.assertIn("https://", str(ctx.exception))

    def test_tunnel_url_rejects_trycloudflare(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            CloudflareTunnel(
                "http://127.0.0.1:9000",
                tunnel_name="t",
                tunnel_url="https://my-tunnel.trycloudflare.com",
            )
        self.assertIn("trycloudflare.com", str(ctx.exception))

    def test_parse_tunnel_list_json(self) -> None:
        payload = json.dumps([{"id": TUNNEL_UUID, "name": "my-tunnel"}])
        parsed = _parse_tunnel_list_json(payload)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[0]["name"], "my-tunnel")
        self.assertEqual(parsed[0]["id"], TUNNEL_UUID)

    def test_lookup_named_tunnel_resolves_credentials_path(self) -> None:
        creds = _credentials_path_for_tunnel_id(TUNNEL_UUID)
        with patch("deepseek_cursor_proxy.tunnel.subprocess.run") as run_mock:
            run_result = MagicMock()
            run_result.returncode = 0
            run_result.stdout = mock_tunnel_list_json([{"id": TUNNEL_UUID, "name": "my-tunnel"}])
            run_mock.return_value = run_result
            with patch.object(Path, "is_file", return_value=True):
                info = lookup_named_tunnel("cloudflared", "my-tunnel")
        self.assertEqual(info.tunnel_id, TUNNEL_UUID)
        self.assertEqual(info.credentials_file, creds)

    def test_lookup_named_tunnel_missing_credentials_raises(self) -> None:
        with patch("deepseek_cursor_proxy.tunnel.subprocess.run") as run_mock:
            run_result = MagicMock()
            run_result.returncode = 0
            run_result.stdout = mock_tunnel_list_json([{"id": TUNNEL_UUID, "name": "my-tunnel"}])
            run_mock.return_value = run_result
            with patch.object(Path, "is_file", return_value=False):
                with self.assertRaises(RuntimeError) as ctx:
                    lookup_named_tunnel("cloudflared", "my-tunnel")
        self.assertIn("credentials", str(ctx.exception))

    def test_cloudflared_not_found_raises_runtime_error(self) -> None:
        with patch("deepseek_cursor_proxy.tunnel.shutil.which", return_value=None):
            tunnel = CloudflareTunnel(
                "http://127.0.0.1:9000",
                tunnel_name="test-tunnel",
                tunnel_url="https://proxy.example.com",
            )
            with self.assertRaises(RuntimeError) as ctx:
                tunnel.start()
            self.assertIn("cloudflared", str(ctx.exception))

    def test_cloudflared_connection_registered_returns_tunnel_url(self) -> None:
        _patch_tunnel_lookup(self)
        with mock_cloudflared_start():
            tunnel = CloudflareTunnel(
                "http://127.0.0.1:9000",
                tunnel_name="my-tunnel",
                tunnel_url="https://proxy.example.com",
            )
            url = tunnel.start()
        self.assertEqual(url, "https://proxy.example.com")

    def test_tunnel_config_uses_tunnel_uuid_and_credentials_json(self) -> None:
        _patch_tunnel_lookup(self)
        with mock_cloudflared_start() as mocks:
            tunnel = CloudflareTunnel(
                "http://127.0.0.1:9000",
                tunnel_name="my-tunnel",
                tunnel_url="https://proxy.example.com",
            )
            tunnel.start()
        content = mocks.write_text.call_args[0][0]
        self.assertIn(f"tunnel: {TUNNEL_UUID}", content)
        self.assertIn(f"credentials-file: {TUNNEL_INFO.credentials_file}", content)
        self.assertNotIn("cert.pem", content)

    def test_startup_timeout_when_no_connection_line(self) -> None:
        _patch_tunnel_lookup(self)
        with mock_cloudflared_start(select_ready=False):
            tunnel = CloudflareTunnel(
                "http://127.0.0.1:9000",
                tunnel_name="my-tunnel",
                tunnel_url="https://proxy.example.com",
                startup_timeout=0.2,
            )
            with self.assertRaises(RuntimeError) as ctx:
                tunnel.start()
        self.assertIn("Timed out", str(ctx.exception))

    def test_monitor_restarts_on_exit(self) -> None:
        tunnel = CloudflareTunnel(
            "http://127.0.0.1:9000",
            tunnel_name="my-tunnel",
            tunnel_url="https://proxy.example.com",
            max_restarts=2,
            probe_public_health=False,
        )
        tunnel.process = MagicMock()
        tunnel.process.poll.return_value = 1
        with patch.object(tunnel, "_restart_tunnel") as restart_mock:
            with patch.object(tunnel._monitor_stop, "wait", side_effect=[False, False, True]) as wait_mock:
                tunnel._monitor_loop()
        restart_mock.assert_called_once()
        self.assertEqual(wait_mock.call_args_list[1].kwargs["timeout"], 2)
        self.assertFalse(tunnel.dead)

    def test_monitor_marks_dead_after_max_restarts(self) -> None:
        tunnel = CloudflareTunnel(
            "http://127.0.0.1:9000",
            tunnel_name="my-tunnel",
            tunnel_url="https://proxy.example.com",
            max_restarts=1,
        )
        tunnel.process = MagicMock()
        tunnel.process.poll.return_value = 1
        with patch.object(tunnel, "_restart_tunnel"):
            with patch.object(tunnel._monitor_stop, "wait", side_effect=[False, False, False]):
                tunnel._monitor_loop()
        self.assertTrue(tunnel.dead)

    def test_named_tunnel_not_found_raises_runtime_error(self) -> None:
        with patch("deepseek_cursor_proxy.tunnel.shutil.which", return_value="/x/cloudflared"):
            with patch(
                "deepseek_cursor_proxy.tunnel.lookup_named_tunnel",
                side_effect=RuntimeError("Named tunnel 'missing-tunnel' not found."),
            ):
                tunnel = CloudflareTunnel(
                    "http://127.0.0.1:9000",
                    tunnel_name="missing-tunnel",
                    tunnel_url="https://proxy.example.com",
                )
                with self.assertRaises(RuntimeError) as ctx:
                    tunnel.start()
        self.assertIn("not found", str(ctx.exception))


class LookupTunnelListTests(unittest.TestCase):
    def test_lookup_raises_when_name_absent(self) -> None:
        with patch("deepseek_cursor_proxy.tunnel.subprocess.run") as run_mock:
            run_result = MagicMock()
            run_result.returncode = 0
            run_result.stdout = mock_tunnel_list_json([{"id": TUNNEL_UUID, "name": "other-tunnel"}])
            run_mock.return_value = run_result
            with self.assertRaises(RuntimeError):
                lookup_named_tunnel("cloudflared", "missing")


if __name__ == "__main__":
    unittest.main()
