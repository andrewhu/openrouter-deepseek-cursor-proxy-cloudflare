<h1>Cloudflare + OpenRouter DeepSeek Cursor Proxy</h1>

## Purpose

This is an **opinionated, single-user bridge** that lets you run **DeepSeek V4 Pro in Cursor agent mode** with **extra-high thinking** (`reasoning.effort: xhigh`) without hitting DeepSeek’s `reasoning_content` tool-call errors.

It is **not** a general-purpose API gateway. The upstream model, provider routing, thinking effort, reasoning repair, recovery behavior, and Cursor thinking UI are **fixed in code** for one workflow: **Cursor → your Cloudflare Tunnel → this proxy → OpenRouter → DeepSeek’s first-party provider only**.

If you need multiple models, alternate providers, lower reasoning effort, or a shared multi-tenant relay, use [OpenRouter](https://openrouter.ai/) or another proxy directly instead of forking this repo.

## Who this is for

The design matches a **personal always-on setup** (for example, a Mac Mini on home Wi‑Fi). See [Always-on Mac Mini (launchd)](#always-on-mac-mini-launchd) to run the proxy at boot.

- **One person** uses the proxy (your machine, your tunnel hostname).
- **One OpenRouter API key** — Cursor sends it as the Bearer token; the proxy checks `sha256(key)` so your public tunnel URL is not an open relay for arbitrary keys. This is a lightweight personal-use gate, not a production security boundary.
- **One model in Cursor:** `deepseek-v4-pro` (mapped upstream to `deepseek/deepseek-v4-pro`).
- **Production path always uses a named Cloudflare Tunnel** on your own domain (Cursor cannot call `localhost`). `--local` exists only for unit tests and manual debugging.

## Hardcoded behavior (not configurable)

| Area | Fixed choice |
|------|----------------|
| Upstream | [OpenRouter](https://openrouter.ai/api/v1) only (not `api.deepseek.com`) |
| Model | `deepseek/deepseek-v4-pro` (Cursor id: `deepseek-v4-pro`) |
| Provider | `provider.only: ["deepseek"]` — OpenRouter’s first-party DeepSeek route; **no third-party fallbacks** |
| Thinking | Always on; `reasoning.effort: xhigh` on every upstream request |
| Missing `reasoning_content` | Always repair from SQLite cache; if still missing, **recover** (truncate history + notice) |
| Cursor UI | Always mirror thinking into collapsible `<details><summary>Thinking</summary>…</details>` blocks |
| Tunnel | Named tunnel `deepseek-proxy` + your `tunnel_url` in config |
| Config surface | `proxy_api_key_hash`, `tunnel_url`, bind `host`/`port`, `verbose` — see generated `~/.deepseek-cursor-proxy/config.yaml` |

## What it does

- Routes every request through OpenRouter to **DeepSeek V4 Pro** with **xhigh** reasoning on the **DeepSeek provider only**.
- Injects `reasoning_content` into outgoing tool-call turns (Cursor omits it), restoring prior reasoning from regular and streamed responses via a local SQLite cache.
- Shows thinking tokens in Cursor as collapsible Markdown thinking blocks.
- Starts **cloudflared** on normal launch so Cursor reaches the proxy over HTTPS on your domain.
- Applies small protocol shims (tools/function_call conversion, content flattening, etc.) so agent mode stays stable.

## Why this exists

Cursor + DeepSeek thinking mode breaks on multi-step tool calls when prior `reasoning_content` is not sent back:

<img src="assets/error_400.png" width="600" alt="Error 400 - reasoning_content must be passed back">

```txt
⚠️ Connection Error
Provider returned error:
{
  "error": {
    "message": "The reasoning_content in the thinking mode must be passed back to the API.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_request_error"
  }
}
```

This proxy caches reasoning from upstream responses and patches it into later requests so agent loops can continue.

## Usage

### Step 1: Set up Cloudflare Tunnel

Cursor blocks non-public API URLs such as `localhost`, so the proxy needs a public HTTPS URL. [Cloudflare Tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) expose the local proxy with a stable hostname on your domain.

One-time setup:

```bash
# Install cloudflared
brew install cloudflared

# Authenticate with Cloudflare (opens browser)
cloudflared tunnel login

# Create a named tunnel (name is fixed in the proxy)
cloudflared tunnel create deepseek-proxy

# Route your domain to the tunnel
cloudflared tunnel route dns deepseek-proxy proxy.yourdomain.com
```

### Step 2: Install and start the proxy

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/yxlao/deepseek-cursor-proxy.git
cd deepseek-cursor-proxy
uv run deepseek-cursor-proxy
```

On first run the proxy creates `~/.deepseek-cursor-proxy/config.yaml` and exits until `proxy_api_key_hash` and `tunnel_url` are set. Edit it to match your install (see [`config.example.yaml`](config.example.yaml) for the full template):

```yaml
proxy_api_key_hash: "<sha256-of-your-openrouter-api-key>"
tunnel_url: https://proxy.yourdomain.com
```

Generate the hash without putting your key in shell history:

```bash
python -c "import hashlib, getpass; k=getpass.getpass('OpenRouter key: '); print(hashlib.sha256(k.encode()).hexdigest())"
```

Store the hash in `~/.deepseek-cursor-proxy/config.yaml` only. Put the real OpenRouter key in Cursor settings. Keeping the raw key out of the repo is still recommended because it avoids accidental cleanup work, even when the key has a low monthly spend cap. Repo-local files such as `dev.config.yaml` are gitignored and excluded from Cursor AI context, but the home-directory config is the canonical location.

Only that key is accepted; other Bearer tokens get HTTP 401.

**Do not use TryCloudflare quick tunnels** (`*.trycloudflare.com`) — they do not support Server-Sent Events (SSE), which Cursor streaming needs. Use a named tunnel on your own domain.

Optional: add [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/) on the hostname if you want an extra gate in front of the tunnel.

The proxy writes ingress config under `~/.deepseek-cursor-proxy/tunnels/deepseek-proxy.yml` using credentials from `~/.cloudflared/<tunnel-uuid>.json`.

Override the public URL without editing config:

```bash
deepseek-cursor-proxy --tunnel-url https://proxy.yourdomain.com
```

**Local / dev only** (no tunnel — unit tests and curl; Cursor cannot use localhost):

```bash
deepseek-cursor-proxy --local --verbose
```

First-run artifacts:

- `~/.deepseek-cursor-proxy/config.yaml`
- `~/.deepseek-cursor-proxy/reasoning_content.sqlite3` (reasoning cache)

Other flags:

```bash
deepseek-cursor-proxy --verbose                # log detailed metadata with truncated message fields
deepseek-cursor-proxy --port 9000              # change local bind port
deepseek-cursor-proxy --trace-dir ./trace-dumps # write sanitized request traces
deepseek-cursor-proxy --clear-reasoning-cache  # wipe reasoning SQLite cache
```

### Always-on Mac Mini (launchd)

Use a **LaunchDaemon** so the proxy (and its `cloudflared` subprocess) start at **boot**, without anyone logging in. The daemon runs as your macOS user so it can read `~/.cloudflared/` and `~/.deepseek-cursor-proxy/`.

```text
Mac reboot → launchd → deepseek-cursor-proxy → cloudflared → Cursor over HTTPS → OpenRouter
```

**Prerequisites** — complete [Step 1](#step-1-set-up-cloudflare-tunnel) and [Step 2](#step-2-install-and-start-the-proxy) first:

1. `cloudflared` installed; tunnel `deepseek-proxy` created and DNS routed.
2. `~/.deepseek-cursor-proxy/config.yaml` has valid `proxy_api_key_hash` and `tunnel_url`.
3. `~/.cloudflared/<tunnel-uuid>.json` exists for the same user you will set as `UserName` in the plist.
4. A manual run succeeds: `uv run deepseek-cursor-proxy` (no `--local`) until logs show **PUBLIC TUNNEL ACTIVE**.

Do **not** run `cloudflared` via `brew services` separately — the proxy starts and monitors `cloudflared` itself; a second tunnel process will conflict.

**Install to a fixed path** (launchd does not use your shell `PATH` for `uv run`):

```bash
export REPO_DIR="$HOME/src/deepseek-cursor-proxy"   # pick a stable location
git clone https://github.com/yxlao/deepseek-cursor-proxy.git "$REPO_DIR"
cd "$REPO_DIR"
uv sync
# Binary used by the service:
#   $REPO_DIR/.venv/bin/deepseek-cursor-proxy
```

**Create the LaunchDaemon plist** from the repo template:

```bash
cp deploy/com.deepseek-cursor-proxy.plist.example ~/com.deepseek-cursor-proxy.plist
```

Edit `~/com.deepseek-cursor-proxy.plist`:

- Replace `YOUR_USER` with your macOS short username (`whoami`).
- Set `ProgramArguments` to `$REPO_DIR/.venv/bin/deepseek-cursor-proxy` (absolute path).
- Confirm log paths under `/Users/YOUR_USER/Library/Logs/deepseek-cursor-proxy/`.

**Enable the service:**

```bash
mkdir -p ~/Library/Logs/deepseek-cursor-proxy

sudo cp ~/com.deepseek-cursor-proxy.plist /Library/LaunchDaemons/com.deepseek-cursor-proxy.plist
sudo chown root:wheel /Library/LaunchDaemons/com.deepseek-cursor-proxy.plist
sudo chmod 644 /Library/LaunchDaemons/com.deepseek-cursor-proxy.plist

sudo launchctl bootstrap system /Library/LaunchDaemons/com.deepseek-cursor-proxy.plist
sudo launchctl enable system/com.deepseek-cursor-proxy
sudo launchctl kickstart -k system/com.deepseek-cursor-proxy
```

**Verify:**

```bash
sudo launchctl print system/com.deepseek-cursor-proxy
tail -f ~/Library/Logs/deepseek-cursor-proxy/stderr.log
curl -sS -H "Authorization: Bearer <your-openrouter-api-key>" "https://proxy.yourdomain.com/v1/healthz"
```

**Stop or remove the service** (maintenance):

```bash
sudo launchctl bootout system /Library/LaunchDaemons/com.deepseek-cursor-proxy.plist
sudo rm /Library/LaunchDaemons/com.deepseek-cursor-proxy.plist
```

**Updates** after `git pull`:

```bash
cd "$REPO_DIR" && uv sync
sudo launchctl kickstart -k system/com.deepseek-cursor-proxy
```

**Config changes** — edit `~/.deepseek-cursor-proxy/config.yaml`, then `kickstart` (no plist edit unless you change bind `port` or `REPO_DIR`).

**Troubleshooting:**

- `cloudflared is not installed or is not on PATH` — ensure `PATH` in the plist includes `/opt/homebrew/bin` (Apple Silicon) or `/usr/local/bin` (Intel Homebrew).
- Tunnel auth errors — run `cloudflared tunnel login` as `YOUR_USER`, not as root.
- Service exits immediately — check `stderr.log`; confirm `proxy_api_key_hash` and `tunnel_url` in config.

The daemon runs as your user (not root) so Cloudflare credentials stay in your home directory. See [SECURITY.md](SECURITY.md) for the threat model.

If you prefer the proxy to start only **after you log in**, use a LaunchAgent in `~/Library/LaunchAgents/` instead of a LaunchDaemon; boot-before-login requires the daemon approach above.

### Step 3: Add the Cursor custom model

In Cursor → Models → Add custom model:

| Field | Value |
|-------|--------|
| Model | `deepseek-v4-pro` |
| API Key | Your [OpenRouter](https://openrouter.ai/settings/keys) key (same key you hashed above) |
| Base URL | `https://proxy.yourdomain.com/v1` (your tunnel host + `/v1`) |

The proxy maps `deepseek-v4-pro` → OpenRouter `deepseek/deepseek-v4-pro` with xhigh reasoning and DeepSeek-only routing. There is no `base_url` or model override in config.

<img src="assets/cursor_config.png" width="600" alt="Cursor settings for DeepSeek through the proxy">

Toggle the custom API:

- macOS: `Cmd+Shift+0`
- Windows/Linux: `Ctrl+Shift+0`

### Step 4: Use DeepSeek in Cursor

Select `deepseek-v4-pro` and use chat or agent mode as usual.

<img src="assets/cursor_chat.png" width="480" alt="Chatting with DeepSeek in Cursor">

## How it works

```text
Cursor  →  Cloudflare Tunnel  →  proxy  →  OpenRouter  →  DeepSeek provider only
                                              deepseek/deepseek-v4-pro
                                              reasoning.effort: xhigh
```

- **Core fix:** Thinking-mode tool calls require the full multi-round `reasoning_content` chain on later turns. Cursor drops it → 400. The proxy stores reasoning from responses (OpenRouter’s `reasoning` field is normalized to `reasoning_content`) and patches missing blocks before forwarding.
- **Cache scopes:** Keys combine a hash of the conversation prefix (roles, content, tool calls — not `reasoning_content`), upstream model, fixed reasoning settings, and API-key hash so parallel chats do not collide.
- **Context caching:** No synthetic thread IDs or timestamps; restored `reasoning_content` is the exact upstream string.
- **Other shims:** `functions`/`function_call` → `tools`/`tool_choice`, strip mirrored thinking blocks from assistant content, flatten multipart content, mirror reasoning into Cursor details blocks.
- **Recovery:** If reasoning is still missing after cache repair, history is truncated to the latest user turn (plus leading system messages) and a short recovery notice is prepended.

## Development

See [SECURITY.md](SECURITY.md) for the project’s personal-use threat model and lightweight repo hygiene.

Unit tests (proxy uses `--local`):

```bash
uv run python -m unittest discover -s tests
```

Pre-commit:

```bash
uv sync --dev
uv run pre-commit run --all-files
```

### Smoke tests (OpenRouter, manual only)

Hits the real OpenRouter API and bills usage. Not part of CI or the default unit suite.

```bash
export RUN_LIVE_OPENROUTER_TESTS=1
export LIVE_OPENROUTER_KEY=sk-or-...   # or OPENROUTER_API_KEY
uv run python -m tests.smoke
```

Uses `sha256(your_key)` as `proxy_api_key_hash`, matching production. Covers auth, wrong-bearer rejection, a short completion, and the tool-call reasoning repair loop.

## Debugging

```bash
deepseek-cursor-proxy --local --verbose --trace-dir ./trace-dumps
```

Clear the reasoning cache:

```bash
deepseek-cursor-proxy --clear-reasoning-cache
```

Alternate config path (local dev only — copy from [`config.example.yaml`](config.example.yaml); avoid committing real hashes to the repo):

```bash
cp config.example.yaml dev.config.yaml
# edit dev.config.yaml with your hash and tunnel_url
deepseek-cursor-proxy --config ./dev.config.yaml
```
