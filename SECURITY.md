# Security Policy

This project is a **single-user local proxy** that exposes an OpenAI-compatible API to Cursor over a Cloudflare Tunnel and forwards requests to OpenRouter. It is designed for personal development use, not multi-tenant production hosting.

## Supported versions

| Version | Supported |
|---------|-----------|
| latest on `main` | yes |
| older releases | best effort |

## Reporting a vulnerability

If you find a security issue, please **open a private security advisory** on GitHub (Security → Advisories → Report a vulnerability) or open a GitHub issue with minimal reproduction details if private reporting is unavailable.

Avoid pasting real API keys, tunnel credentials, or conversation content into public issues. Even for a spend-capped key, keeping them out of public threads avoids cleanup noise.

## Threat model

**In scope**

- Accidental credential commits (OpenRouter keys, Cloudflare tunnel JSON)
- Unauthorized use of a public tunnel hostname
- Leakage of prompts or reasoning content via logs, traces, or local cache

**Out of scope (by design for personal use)**

- Treating OpenRouter API-key disclosure as catastrophic; use a spend cap and rotate the key if needed
- Multi-user isolation on a shared proxy instance
- Protection against a compromised local machine
- Hardening against determined attackers with network access to your laptop

## Recommended credential placement

The proxy does not try to be a secret-management system. These defaults keep setup simple and reduce accidental repo churn.

| Credential | Recommended place | Avoid committing |
|--------|------------|------------------|
| OpenRouter API key | Cursor model settings | Repo, config files, shell history |
| `proxy_api_key_hash` | `~/.deepseek-cursor-proxy/config.yaml` | Committed YAML (use [`config.example.yaml`](config.example.yaml) as template) |
| Cloudflare tunnel credentials | `~/.cloudflared/<tunnel-uuid>.json` | Repo or project directory |
| Smoke-test key | Shell env (`LIVE_OPENROUTER_KEY`) | `.env` committed to git |

Generate the config hash without echoing the key to shell history:

```bash
python -c "import hashlib, getpass; k=getpass.getpass('OpenRouter key: '); print(hashlib.sha256(k.encode()).hexdigest())"
```

## Runtime controls

- **Proxy auth:** Requests must present a Bearer token whose SHA-256 matches `proxy_api_key_hash`. Other tokens receive HTTP 401. This is meant to prevent casual public-relay use of the tunnel, not to secure a shared service.
- **Bind address:** The proxy listens on loopback only (`127.0.0.1`, `localhost`, or `::1`).
- **Upstream:** Hardcoded to OpenRouter; there is no user-configurable upstream URL.
- **File permissions:** Config, reasoning cache, and trace files are created with restrictive permissions (`0o600` / `0o700`).

Optional extra gate for a public hostname: [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/) on the tunnel hostname.

## Local data sensitivity

These paths may contain conversation content or code from Cursor sessions:

- `~/.deepseek-cursor-proxy/reasoning_content.sqlite3` — reasoning cache (required for tool-call repair); stores plaintext reasoning for repair
- `./trace-dumps/` (or `--trace-dir`) — structured request traces with **sanitized** bodies by default: message summaries (lengths and SHA-256 fingerprints), not full prompt text; upstream/cursor response bodies are hashes and byte counts only
- stdout with `--verbose` — request/response JSON logged with **truncated** message fields (content, reasoning, tool arguments capped at 200 characters); Authorization is never logged in full

Treat them like chat history. Do not share or back them up to untrusted locations.

## Repo hygiene

The following are **gitignored** and listed in **`.cursorignore`** so they are not committed or sent to coding agents:

- `config.yaml`, `dev.config.yaml`, `*.local.yaml`
- `.env`, `.env.*`
- `trace-dumps/`, `.deepseek-cursor-proxy/`

Pre-commit runs **[Gitleaks](https://github.com/gitleaks/gitleaks)** as a cheap guardrail against accidental credential commits. Install hooks:

```bash
uv sync --dev
uv run pre-commit install
```

Run manually:

```bash
uv run pre-commit run gitleaks --all-files
```

Skip in an emergency (not recommended):

```bash
SKIP=gitleaks git commit -m "message"
```

## Security-related dependencies

Keep dependencies updated (`uv sync`, review `uv.lock`). CI runs unit tests and pre-commit on pushes and pull requests; it does not use live API keys.
