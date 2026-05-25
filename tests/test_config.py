from __future__ import annotations

import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deepseek_cursor_proxy.config import (
    CURSOR_MODEL_ID,
    DEFAULT_PORT,
    DEFAULT_VERBOSE,
    ProxyConfig,
    default_config_path,
    default_reasoning_content_path,
    validate_proxy_api_key_hash,
)


class ConfigTests(unittest.TestCase):
    def test_default_paths_live_in_visible_user_app_directory(self) -> None:
        home = Path("/tmp/home")

        with patch("deepseek_cursor_proxy.config.Path.home", return_value=home):
            self.assertEqual(default_config_path(), home / ".deepseek-cursor-proxy" / "config.yaml")
            self.assertEqual(
                default_reasoning_content_path(),
                home / ".deepseek-cursor-proxy" / "reasoning_content.sqlite3",
            )
            self.assertIsNone(ProxyConfig().tunnel_url)
            self.assertIsNone(ProxyConfig().trace_dir)

    def test_missing_default_config_file_is_populated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)

            with patch("deepseek_cursor_proxy.config.Path.home", return_value=home):
                ProxyConfig.from_file(config_path=None)
                config_path = default_config_path()

            config_text = config_path.read_text(encoding="utf-8")

            self.assertTrue(config_path.exists())
            self.assertIn("proxy_api_key_hash:", config_text)
            self.assertIn("tunnel_url:", config_text)
            self.assertIn("host: 127.0.0.1", config_text)
            self.assertIn(f"port: {DEFAULT_PORT}", config_text)
            self.assertIn(CURSOR_MODEL_ID, config_text)
            self.assertNotIn("tunnel_name:", config_text)
            self.assertNotIn("base_url:", config_text)
            self.assertNotIn("model:", config_text)
            self.assertNotIn("thinking:", config_text)
            self.assertNotIn("missing_reasoning_strategy:", config_text)
            self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o600)

    def test_missing_explicit_config_file_is_not_populated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing.yaml"

            config = ProxyConfig.from_file(config_path=config_path)

            self.assertFalse(config_path.exists())
            self.assertEqual(config.port, DEFAULT_PORT)

    def test_loads_config_from_user_yaml_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "proxy_api_key_hash: " + ("a" * 64),
                        "port: 9100",
                        "host: 127.0.0.1",
                        "verbose: true",
                        "tunnel_url: https://proxy.example.com",
                        # legacy keys are ignored
                        "model: deepseek-v4-flash",
                        "thinking: disabled",
                        "missing_reasoning_strategy: reject",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_file(config_path=config_path)

        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 9100)
        self.assertTrue(config.verbose)
        self.assertEqual(config.tunnel_url, "https://proxy.example.com")
        self.assertEqual(config.proxy_api_key_hash, "a" * 64)

    def test_proxy_api_key_hash_is_normalized_to_lowercase(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("proxy_api_key_hash: " + ("A" * 64) + "\n", encoding="utf-8")

            config = ProxyConfig.from_file(config_path=config_path)

        self.assertEqual(config.proxy_api_key_hash, "a" * 64)

    def test_invalid_config_values_fall_back_to_defaults(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "port: nope",
                        "verbose: maybe",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_file(config_path=config_path)

        self.assertEqual(config.port, DEFAULT_PORT)
        self.assertEqual(config.verbose, DEFAULT_VERBOSE)

    def test_tunnel_url_empty_or_whitespace_is_none(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text('tunnel_url: "   "\n', encoding="utf-8")
            config = ProxyConfig.from_file(config_path=config_path)
        self.assertIsNone(config.tunnel_url)

    def test_invalid_yaml_config_raises_value_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                ProxyConfig.from_file(config_path=config_path)

    def test_process_environment_does_not_override_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("verbose: false\n", encoding="utf-8")

        with patch.dict(
            "os.environ",
            {
                "PROXY_VERBOSE": "true",
                "DEEPSEEK_CURSOR_PROXY_CONFIG_PATH": "/ignored.yaml",
            },
            clear=True,
        ):
            config = ProxyConfig.from_file(config_path=config_path)
            self.assertEqual(
                dict(os.environ),
                {
                    "PROXY_VERBOSE": "true",
                    "DEEPSEEK_CURSOR_PROXY_CONFIG_PATH": "/ignored.yaml",
                },
            )

        self.assertFalse(config.verbose)

    def test_validate_proxy_api_key_hash_requires_64_hex(self) -> None:
        with self.assertRaises(ValueError):
            validate_proxy_api_key_hash("not-a-hash")
        with self.assertRaises(ValueError):
            validate_proxy_api_key_hash(None)
        validate_proxy_api_key_hash("a" * 64)


if __name__ == "__main__":
    unittest.main()
