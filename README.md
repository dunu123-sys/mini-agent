# mini-agent

A minimal coding-agent harness plus a live web UI, built to run a **small local model** (Qwen3.5 4B/9B on a llama.cpp server) as a usable coding assistant.

The premise: a 4B model cannot afford a 31 KB prompt. Big agent harnesses ship a large system prompt and a dozen verbose tool schemas — around 7,800 tokens that a small model must prefill on *every* turn. On a consumer GPU that is 50–150 s of waiting before the model says anything, and a tool surface the model can't keep straight.

This harness keeps it to **seven small tools, a ~1.8 KB system prompt, and one loop**:

| | Prompt size | Prefill per turn |
|---|---|---|
| Typical harness | ~31 KB (~7,800 tok) | 50–150 s |
| mini-agent | ~2.4 KB (~600 tok) | 3–8 s |

Measured on an RX 580 8 GB running Qwen3.5 4B Q4_K_M at 32k context.

## What you get

**`mini-agent.py`** — the harness. Tools: `read`, `grep`, `list`, `write`, `append`, `edit`, `bash`. Persistent JSON sessions, streaming output, JSON event mode, and context compaction that trims tool results first and then summarizes.

**`mini-agent-web.py`** — a single-page UI over the same agent:

- reasoning and answer stream token by token;
- tool calls render as collapsible blocks with their output;
- sidebar of sessions with token meters, compaction state and per-session context size;
- controls for the model profile (4B/32k vs 9B/16k), reasoning budget, and a model restart that clears the degenerate-repetition state small quantized models fall into;
- one config file drives both the page and the agent, so they can never disagree about the context budget.

## Quick start

Requires Python 3.9+ and an OpenAI-compatible chat endpoint (llama.cpp's `llama-server` is what this was built against).

```bat
run.cmd
```

That copies `config.example.json` to `config.json` on first run and serves the UI at <http://127.0.0.1:8790>.

Point it at your server in `config.json`:

```json
"agent": { "server": "http://127.0.0.1:8099/v1", "model": "qwen35-4b", "ctx_size": 32768 }
```

CLI use, no server needed for the page:

```bash
python mini-agent.py "list the python files here and summarize each"
python mini-agent.py --session work "add a docstring to main.py"
python mini-agent.py --session work --compact
python mini-agent.py --list-sessions
python mini-agent.py --json "..."      # machine-readable events
```

## Configuration

`config.json` is the single source of truth (`config.example.json` documents every key):

| Section | What it controls |
|---|---|
| `agent` | endpoint, model alias, context size, step and token budgets, temperature, tool-call retry |
| `tools` | result truncation, read windowing/outline thresholds, grep limits, bash timeout and kill-tree |
| `compaction` | auto-compact trigger, how many messages/tokens to keep, tool-result trimming, digest size |
| `prompt` | system prompt file, optional memory/user files injected into the system message |
| `ui` | port, sessions directory, title, poll interval |

Paths are resolved as: absolute if absolute, otherwise relative to the script folder. Set `MINI_AGENT_ROOT` to keep `state/`, `models/` and `logs/` outside the checkout.

## Design notes

- **Tool results are budgeted, not dumped.** Every result passes through a cap with head/tail weighting, and `read` supports ranges and outlines so the model asks for a window instead of a whole file.
- **Compaction trims before it summarizes.** Old tool output is shrunk first; only then does the harness call the model for a digest. Cheaper, and it preserves the conversation's shape.
- **Session files are plain JSON** — inspect, diff, or hand-edit them.
- **No framework.** Standard library only: `urllib` for the SSE stream, `http.server` for the page. Nothing to install.

## Not included

Model weights, the launcher scripts for `llama-server`, and any host-specific service wrappers. Bring your own server; this repo is the harness and the UI.

## License

MIT — see `LICENSE`.
