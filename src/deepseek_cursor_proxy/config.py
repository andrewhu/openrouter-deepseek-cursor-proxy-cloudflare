from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

APP_DIR_NAME = ".deepseek-cursor-proxy"
CONFIG_FILE_NAME = "config.yaml"
REASONING_CONTENT_FILE_NAME = "reasoning_content.sqlite3"

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
MISSING = object()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
DEFAULT_UPSTREAM_BASE_URL = "https://openrouter.ai/api/v1"
CURSOR_MODEL_ID = "deepseek-v4-pro"
OPENROUTER_MODEL_ID = "deepseek/deepseek-v4-pro"
OPENROUTER_PROVIDER_ONLY = ["deepseek"]
REASONING_EFFORT = "xhigh"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_VERBOSE = False
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_MAX_REQUEST_BODY_BYTES = 20 * 1024 * 1024
DEFAULT_REQUEST_QUEUE_SIZE = 128
DEFAULT_MAX_CONCURRENT_REQUESTS = 32
DEFAULT_TUNNEL_NAME = "deepseek-proxy"
DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
DEFAULT_REASONING_CACHE_MAX_ROWS = 100_000

DEFAULT_CONFIG_HEADER = "# This file was created automatically at ~/.deepseek-cursor-proxy/config.yaml."
DEFAULT_CONFIG_TEXT = f"""{DEFAULT_CONFIG_HEADER}
# Cursor sends your OpenRouter API key as Bearer; the proxy checks it against proxy_api_key_hash.
# Upstream is always OpenRouter DeepSeek V4 Pro (model id {CURSOR_MODEL_ID} in Cursor).

proxy_api_key_hash: "<sha256-of-your-openrouter-api-key>"
tunnel_url: https://proxy.yourdomain.com

host: {DEFAULT_HOST}
port: {DEFAULT_PORT}
verbose: {str(DEFAULT_VERBOSE).lower()}
"""


def default_app_dir() -> Path:
    return Path.home() / APP_DIR_NAME


def default_config_path() -> Path:
    return default_app_dir() / CONFIG_FILE_NAME


def default_reasoning_content_path() -> Path:
    return default_app_dir() / REASONING_CONTENT_FILE_NAME


def populate_default_config_file(config_path: Path) -> None:
    config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_path.parent.chmod(0o700)
    config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    config_path.chmod(0o600)


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        return {}

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config at {config_path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return dict(loaded)


def resolve_config_path(config_path: str | Path | None) -> Path:
    return Path(config_path or default_config_path()).expanduser()


def setting_value(settings: Mapping[str, Any], key: str) -> Any:
    return settings.get(key, MISSING)


def as_str(value: Any, default: str) -> str:
    if value is MISSING or value is None:
        return default
    return str(value)


def as_optional_str(value: Any) -> str | None:
    if value is MISSING or value is None:
        return None
    stripped = str(value).strip()
    return stripped if stripped else None


def as_bool(value: Any, default: bool) -> bool:
    if value is MISSING or value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def as_int(value: Any, default: int) -> int:
    if value is MISSING or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_openrouter_upstream(base_url: str) -> bool:
    return "openrouter.ai" in base_url


def is_loopback_host(host: str) -> bool:
    return host.strip().lower() in LOOPBACK_HOSTS


def validate_proxy_api_key_hash(value: str | None) -> None:
    if value is None:
        raise ValueError("proxy_api_key_hash is required")
    normalized = value.strip().lower()
    if len(normalized) != 64 or not all(character in "0123456789abcdef" for character in normalized):
        raise ValueError("proxy_api_key_hash must be a 64-character lowercase hex SHA-256 digest")


@dataclass(frozen=True)
class ProxyConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL
    proxy_api_key_hash: str | None = None
    verbose: bool = DEFAULT_VERBOSE
    tunnel_url: str | None = None
    trace_dir: Path | None = None

    @classmethod
    def from_file(
        cls: type[ProxyConfig],
        config_path: str | Path | None = None,
    ) -> "ProxyConfig":
        settings, _resolved_config_path = settings_from_config(config_path)
        proxy_api_key_hash = as_optional_str(setting_value(settings, "proxy_api_key_hash"))
        if proxy_api_key_hash is not None:
            proxy_api_key_hash = proxy_api_key_hash.lower()

        return cls(
            host=as_str(
                setting_value(settings, "host"),
                DEFAULT_HOST,
            ),
            port=as_int(
                setting_value(settings, "port"),
                DEFAULT_PORT,
            ),
            proxy_api_key_hash=proxy_api_key_hash,
            verbose=as_bool(
                setting_value(settings, "verbose"),
                DEFAULT_VERBOSE,
            ),
            tunnel_url=as_optional_str(setting_value(settings, "tunnel_url")),
        )


def settings_from_config(
    config_path: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    resolved_config_path = resolve_config_path(config_path)
    if config_path is None and not resolved_config_path.exists():
        populate_default_config_file(resolved_config_path)
    return load_config_file(resolved_config_path), resolved_config_path
