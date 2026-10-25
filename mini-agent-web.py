#!/usr/bin/env python
"""
mini-agent-web — a tiny local web UI for mini-agent.py.

Serves a single page that lets you type a task, pick a working directory, and
watch the agent work in real time: reasoning streams token by token, tool calls
and their output are collapsible blocks, and sessions can be created, continued,
reset and compacted.

Run:
    python mini-agent-web.py
    -> http://127.0.0.1:8790

API
    GET  /api/status          model up?, busy?
    GET  /api/dirs            candidate working directories
    GET  /api/sessions        list saved sessions
    POST /api/session/new     {id, cwd}          -> create/clear
    POST /api/session/reset   {id}               -> wipe history
    POST /api/session/compact {id, server, model}-> compact history
    POST /api/run             {task, cwd, session, steps} -> NDJSON event stream
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_lib

CFG = config_lib.load()
CONFIG_PATH = config_lib.CONFIG_PATH
def _agent_path():
    """Which agent file this page drives: production ships as mini-agent.py,
    the staging folder keeps mini-agent2.py next to the page."""
    want = (CFG.get("agent") or {}).get("script") or "mini-agent.py"
    p = want if os.path.isabs(want) else os.path.join(HERE, want)
    for cand in (p, os.path.join(HERE, "mini-agent.py"), os.path.join(HERE, "mini-agent2.py")):
        if os.path.isfile(cand):
            return cand
    return p


AGENT = _agent_path()
PORT = int(CFG["ui"]["port"])
STATE_DIR = CFG["ui"]["sessions_dir"]

MODEL_RESTART_LOCK = threading.Lock()   # one stop+load sequence at a time

STATE = {"busy": False, "started": 0.0, "task": "", "cwd": "", "session": ""}
LOCK = threading.Lock()


_META = {}


# Runtime root (state/, logs/, models/, sessions/). Override with
# MINI_AGENT_ROOT when the data tree lives outside the checkout.
ROOT = os.environ.get("MINI_AGENT_ROOT") or HERE

_LOAD_ERR = {"key": None, "text": ""}


def _load_error():
    """One line explaining the last failed load, read from the tail of
    llama-server.err.log - a VRAM refusal is the common case. Read-only:
    it never unloads anything. Cached on (mtime, size) because /api/status
    is polled every few seconds."""
    if os.path.exists(os.path.join(ROOT, "state", "opencodemodel-unloaded")):
        return ""
    p = os.path.join(ROOT, "logs", "llama-server.err.log")
    try:
        st = os.stat(p)
    except OSError:
        return ""
    key = (st.st_mtime, st.st_size)
    if _LOAD_ERR["key"] == key:
        return _LOAD_ERR["text"]
    text = ""
    try:
        with open(p, "rb") as f:
            f.seek(max(0, st.st_size - 6000))
            tail = f.read().decode("utf-8", "replace")
        for line in reversed(tail.splitlines()):
            if "ErrorOutOfDeviceMemory" in line or "out of device memory" in line \
                    or "failed to allocate buffer" in line:
                size = ""
                for l2 in tail.splitlines():
                    m = re.search(r"allocation of size (\d+) failed", l2)
                    if m:
                        size = " (%d MB)" % (int(m.group(1)) // (1024 * 1024))
                        break
                kind = re.search(r"failed to allocate buffer for (\w+) cache", tail)
                what = ("the %s cache" % kind.group(1)) if kind else "device memory"
                text = "no VRAM: could not allocate %s%s - nothing was unloaded" % (what, size)
                break
    except OSError:
        pass
    _LOAD_ERR.update({"key": key, "text": text})
    return text


def _safe_read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _backup(path, cfg):
    """A save from the page overwrites a REAL file (the operator's memory): keep
    one timestamped copy first, never more than needed."""
    if not cfg.get("prompt", {}).get("backup_on_save", True) or not os.path.isfile(path):
        return
    try:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        with open(path, "rb") as src, open(path + ".bak-" + stamp, "wb") as dst:
            dst.write(src.read())
    except OSError:
        pass


def agent_meta():
    """Import the v2 agent once (main() is guarded) to show its real SYSTEM prompt
    and tool list on the Configuration tab - never a hand-copied duplicate."""
    if _META:
        return _META
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mini_agent2_meta", AGENT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _META.update({"system": mod.SYSTEM, "tools": mod.TOOLS,
                      "estimate": mod.estimate_context,
                      "load_system": mod.load_system,
                      "prompt_file": mod.prompt_file})
    except Exception as e:
        _META.update({"system": "", "tools": [], "error": str(e)})
    return _META


def find_python():
    return sys.executable or "python"


def agent_roots():
    """Candidate working dirs. Deduped case- and slash-insensitively: the list
    used to show the home dir twice (home == the literal), which ate
    the sidebar with redundant chips."""
    out, seen = [], set()
    for p in [
        os.path.expanduser("~"),
        ROOT,
        os.path.join(ROOT, "projects"),
        os.path.join(ROOT, "scratch"),
    ]:
        try:
            if not os.path.isdir(p):
                continue
            key = os.path.normcase(os.path.normpath(os.path.abspath(p))).replace("\\", "/")
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p.replace("\\", "/"))
    return out


# ---------------------------------------------------------------- tuning
# The reasoning budget is a SERVER flag (--reasoning-budget), not a request
# field: verified that sending reasoning_budget in the request body changes
# nothing. So the page persists it here and the launcher reads it at start, which
# means applying a new budget costs one model restart.
TUNING_FILE = os.path.join(ROOT, "state", "llm-tuning.cmd")
DEFAULT_BUDGET = 384


def tuning_budget():
    """Budget currently configured for the next/active server start."""
    try:
        with open(TUNING_FILE, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return DEFAULT_BUDGET
    if "REASONING_BUDGET=" not in txt:
        return DEFAULT_BUDGET
    tail = txt.split("REASONING_BUDGET=")[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else DEFAULT_BUDGET


def set_tuning_budget(n):
    """Persist the budget for the next server start; returns the stored value."""
    n = max(0, min(4096, int(n)))
    os.makedirs(os.path.dirname(TUNING_FILE), exist_ok=True)
    config_lib.write_atomic(TUNING_FILE,
        "@echo off\r\nREM written by mini-agent-web.py; read by llama-server-run.cmd\r\n"
        'set "REASONING_BUDGET=%d"\r\n' % n)
    return n


# ---------------------------------------------------------------- model profiles
# One llama-server, one model: the card is 8 GB, so the 4B (2.55 GiB) and the 9B
# (5.24 GiB) can never be resident together. "Using both" therefore means a
# switch that rewrites state\llm-model.cmd and restarts the runner.
# The 9B path is a HARDLINK to Ollama's blob (same volume, 0 bytes): stable name,
# no 5.2 GiB copy, and it survives Ollama re-tagging the model.
PROFILE_FILE = os.path.join(ROOT, "state", "llm-model.cmd")
DEFAULT_PROFILE = "qwen35-4b"
PROFILES = {
    "qwen35-4b": {
        "label": "QWEN3.5 4B - 32k - fast",
        "path": os.path.join(ROOT, "models", "Qwen3.5-4B-Q4_K_M.gguf"),
        "alias": "qwen35-4b",
        "ctx": 32768,
        "kv": "",
    },
    "qwen35-9b": {
        "label": "QWEN3.5 9B - 16k - deeper",
        "path": os.path.join(ROOT, "models", "Qwen3.5-9B-Q4_K_M.gguf"),
        "alias": "qwen35-9b",
        "ctx": 16384,
        "kv": "-ctk q8_0 -ctv q8_0",
    },
}


def loaded_alias(timeout=3):
    """The alias the running runner actually serves (/v1/models), or ''.

    Nothing checked this before: the page said MODEL UP as soon as the port
    answered, so a runner that had been started with the OTHER profile looked
    healthy while every agent call failed with "model not found".
    """
    try:
        with urllib.request.urlopen("http://127.0.0.1:8099/v1/models",
                                    timeout=timeout) as r:
            d = json.load(r)
        ms = d.get("models") or d.get("data") or []
        if ms:
            return str(ms[0].get("name") or ms[0].get("id") or "").strip()
    except Exception:
        pass
    return ""


def current_profile():
    """Which profile the launcher file currently describes."""
    try:
        with open(PROFILE_FILE, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return DEFAULT_PROFILE
    for key in PROFILES:
        if "MODEL_PROFILE=" + key in txt:
            return key
    for key, p in PROFILES.items():
        if "MODEL_ALIAS=" + p["alias"] in txt:
            return key
    return DEFAULT_PROFILE


def set_profile(key):
    """Persist a profile for the next server start. Returns the stored key."""
    p = PROFILES[key]
    os.makedirs(os.path.dirname(PROFILE_FILE), exist_ok=True)
    config_lib.write_atomic(PROFILE_FILE,
        "@echo off\r\nREM written by mini-agent-web.py; read by llama-server-run.cmd\r\n"
        'set "MODEL_PROFILE=%s"\r\n' % key +
        'set "MODEL_PATH=%s"\r\n' % p["path"] +
        'set "MODEL_ALIAS=%s"\r\n' % p["alias"] +
        'set "CTX_SIZE=%d"\r\n' % p["ctx"] +
        'set "KV_FLAGS=%s"\r\n' % p["kv"])
    return key


def _read_text_shared(path):
    """Read a session file with FILE_SHARE_DELETE.

    Python's open() asks for FILE_SHARE_READ|WRITE only, so while the page is
    polling, the agent's atomic os.replace() of that same file fails with
    PermissionError [WinError 5]. Asking for DELETE sharing removes the race
    instead of retrying around it.
    """
    if os.name != "nt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    import ctypes
    from ctypes import wintypes
    GENERIC_READ = 0x80000000
    SHARE_ALL = 0x1 | 0x2 | 0x4            # FILE_SHARE_READ|WRITE|DELETE
    OPEN_EXISTING = 3
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                ctypes.c_void_p]
    k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                             ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    h = k32.CreateFileW(os.fspath(path), GENERIC_READ, SHARE_ALL, None,
                        OPEN_EXISTING, 0, None)
    if h in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_last_error(), "CreateFileW failed", str(path))
    try:
        chunks = []
        buf = ctypes.create_string_buffer(1 << 16)
        n = wintypes.DWORD(0)
        while True:
            if not k32.ReadFile(h, buf, len(buf), ctypes.byref(n), None):
                raise OSError(ctypes.get_last_error(), "ReadFile failed", str(path))
            if not n.value:
                break
            chunks.append(buf.raw[:n.value])
        return b"".join(chunks).decode("utf-8", "replace")
    finally:
        k32.CloseHandle(ctypes.c_void_p(h))


def _retry_os(fn, *a, **kw):
    """Run a metadata operation a few times: on Windows a file that the page is
    reading (its open() does not ask for FILE_SHARE_DELETE) blocks replace/remove
    with WinError 5 for a moment. Retrying is enough; failing is not an option
    because it leaves a duplicate session behind."""
    tries = int(kw.pop("tries", 8))
    last = None
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except OSError as e:
            last = e
            time.sleep(0.03 * (i + 1))
    raise last

def session_file(sid):
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid or "")[:64] or "default"
    return os.path.join(STATE_DIR, safe + ".json")


def session_tokens(d):
    """Exact when the agent measured it (last_usage), estimate otherwise.
    The old pair only ever had the chars/3.6 estimate, which is +-20% on real
    tool traffic - that is what made the CONTEXT meter look wrong."""
    msgs = d.get("messages") or []
    want_ctx = PROFILES.get(current_profile(), {}).get("ctx", 32768)
    lu = d.get("last_usage") or {}
    if lu and int(lu.get("prompt") or 0) > 0 and int(lu.get("ctx") or 0) == want_ctx:
        return int(lu["prompt"]), True
    est = agent_meta().get("estimate")
    tok = est(msgs) if est else sum(int(len(m.get("content") or "") / 3.6) + 8 for m in msgs)
    return tok, False


_SESS_CACHE = {"key": None, "data": None}


def clean_id(sid):
    """Session ids are file names: keep them boring so two different names can
    never land on the same file ('my session' vs 'my_session')."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_.-]", "_", sid or "")[:64] or "default"


def next_session_name():
    """NEW always creates something called 'new' / new-2 / new-3 - never asks
    for a name up front, and never overwrites an existing session."""
    if not os.path.exists(session_file("new")):
        return "new"
    n = 2
    while os.path.exists(session_file("new-%d" % n)):
        n += 1
    return "new-%d" % n


def session_lock_path(sid):
    """Same convention as the agent: <session>.json.lock next to the session."""
    return session_file(sid) + ".lock"


def read_sessions():
    out = []
    if not os.path.isdir(STATE_DIR):
        return out
    for fn in os.listdir(STATE_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            d = json.loads(_read_text_shared(os.path.join(STATE_DIR, fn)))
            tok, exact = session_tokens(d)
            out.append({
                "id": d.get("id", fn[:-5]),
                "title": d.get("title", ""),
                "cwd": d.get("cwd", ""),
                "messages": len(d.get("messages") or []),
                "tokens": tok, "exact": exact,
                "empty": not (d.get("messages") or []),
                "updated": d.get("updated", 0),
            })
        except Exception:
            continue
    out.sort(key=lambda x: x.get("updated", 0), reverse=True)
    return out


def run_agent(args_list, timeout=1800):
    """Run mini-agent.py with args, returning (exit_code, stdout)."""
    prof = PROFILES[current_profile()]
    # The alias matters: llama-server rejects a model id it does not serve, and
    # ctx-size drives the agent's own context estimate, so both follow the profile.
    cmd = [find_python(), AGENT, "--config", CONFIG_PATH,
           "--model", prof["alias"], "--ctx-size", str(prof["ctx"])] + args_list
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return r.returncode, r.stdout, r.stderr


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MINI AGENT</title>
<style>
  :root{
    --bg:#1a1512; --bone:#ead6b0; --dim:#c9b994; --mustard:#d9a86c;
    --brick:#a4382b; --navy:#2b4a36; --navydeep:#1b3322; --iron:#201b18;
    --ok:#7cb342; --panel:rgba(27,51,34,.5);
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;overflow:hidden;background:var(--bg);color:var(--bone);
    font-family:"VT323",Consolas,monospace;font-size:1.1rem;line-height:1.28;
    display:flex;flex-direction:column}

  /* ---- scrollbars: one thin bar per region, themed, never doubled ---- */
  *{scrollbar-width:thin;scrollbar-color:rgba(201,185,148,.45) transparent}
  *::-webkit-scrollbar{width:10px;height:10px}
  *::-webkit-scrollbar-track{background:rgba(0,0,0,.18)}
  *::-webkit-scrollbar-thumb{background:rgba(201,185,148,.38);border:2px solid transparent;
    background-clip:padding-box;border-radius:6px}
  *::-webkit-scrollbar-thumb:hover{background:rgba(217,168,108,.65);background-clip:padding-box}

  /* ---- top bar: one row, wraps only when narrow ---- */
  header{flex:0 0 auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap;
    padding:8px 16px;border-bottom:3px solid var(--brick);background:rgba(26,21,18,.97)}
  .brand{display:flex;align-items:center;gap:11px;margin-right:auto;min-width:0}
  h1{font-family:"Press Start 2P",monospace;font-size:15px;margin:0;letter-spacing:1px}
  .sub{color:var(--dim);font-size:.93rem;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;max-width:46vw}
  .px{font-family:"Press Start 2P",monospace;font-size:9px;letter-spacing:1px}
  .diamond{width:19px;height:19px;background:var(--brick);transform:rotate(45deg);flex:0 0 auto}
  .hgroup{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
  .sep{width:1px;height:22px;background:rgba(201,185,148,.25);margin:0 2px}

  /* ---- shell: header + two columns, each owning its own scroll ---- */
  .main{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:288px minmax(0,1fr);
    gap:13px;padding:13px 16px 14px}
  .side{min-height:0;display:flex;flex-direction:column;gap:11px;overflow-y:auto;
    overflow-x:hidden;padding-right:2px}
  .chat{min-height:0;display:flex;flex-direction:column;border:3px solid var(--bone);
    box-shadow:5px 5px 0 rgba(164,56,43,.42);background:var(--panel);overflow:hidden}

  .panel{border:3px solid var(--bone);box-shadow:4px 4px 0 rgba(164,56,43,.42);
    background:var(--panel);padding:11px;flex:0 0 auto}
  .panel.sessions{flex:1 1 auto;min-height:176px;display:flex;flex-direction:column;
    overflow:hidden}
  #sessinfo{flex:0 0 auto;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .panel.two{display:grid;grid-template-columns:1fr 1.35fr;gap:11px;align-items:start}

  label{display:block;font-family:"Press Start 2P",monospace;font-size:9px;color:var(--mustard);
    margin:0 0 5px;letter-spacing:1px}
  label+label{margin-top:9px}
  input,select,textarea{width:100%;background:var(--iron);border:2px solid var(--dim);
    color:var(--bone);font-family:inherit;font-size:1.02rem;padding:6px 7px;outline:none}
  input:focus,select:focus,textarea:focus{border-color:var(--mustard)}
  button{font-family:"Press Start 2P",monospace;font-size:9px;padding:7px 9px;cursor:pointer;
    border:2px solid var(--dim);background:var(--navydeep);color:var(--bone);letter-spacing:1px}
  button:hover:not(:disabled){border-color:var(--mustard);color:var(--mustard)}
  button:disabled{opacity:.4;cursor:not-allowed}
  button.go{background:var(--navy);border-color:var(--bone);color:var(--bone)}
  button.go:hover:not(:disabled){background:var(--ok);color:var(--iron);border-color:var(--ok)}
  button.stop{background:var(--brick);border-color:var(--bone)}
  button.sm{padding:5px 7px;font-size:8px}
  .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:9px}
  .pill{font-family:"Press Start 2P",monospace;font-size:9px;padding:5px 7px;border:2px solid var(--dim)}
  .pill.on{background:var(--ok);color:var(--iron);border-color:var(--ok)}
  .pill.off{background:var(--brick);color:var(--bone);border-color:var(--brick)}
  .pill.wait{background:var(--mustard);color:var(--iron);border-color:var(--mustard)}
  .meta{color:var(--dim);font-size:.9rem}

  /* ---- sessions: the ONLY growing scroller in the sidebar ---- */
  #sesslist{flex:1 1 auto;min-height:86px;overflow-y:auto;overflow-x:hidden;
    border:2px solid var(--dim);
    background:var(--iron);margin-top:8px}
  #sesslist button{display:block;width:100%;text-align:left;margin:0;border:none;
    border-bottom:1px solid rgba(201,185,148,.14);background:transparent;padding:5px 8px;
    font-family:"VT323",monospace;font-size:1rem;letter-spacing:0}
  #sesslist button:hover{background:var(--navydeep);color:var(--mustard)}
  #sesslist button.sel{background:var(--navy);color:var(--bone)}
  #sesslist button.running::after{content:" \25CF";color:var(--ok)}
  #sesslist .t{color:var(--dim);font-size:.83rem;display:block;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

  /* ---- dirs: wrapping chips, no scrollbar ---- */
  #dirs{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
  #dirs button{font-family:"VT323",monospace;font-size:.88rem;padding:2px 5px;
    background:var(--iron);border:1px solid var(--dim);letter-spacing:0;color:var(--dim);
    max-width:122px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:0 0 auto}
  #dirs button:hover{color:var(--mustard);border-color:var(--mustard)}

  /* ---- transcript: fills the column, composer pinned below ---- */
  #log{flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;
    background:var(--iron);padding:11px 13px}
  .msg{padding:6px 0;border-bottom:1px solid rgba(201,185,148,.1)}
  .msg:last-child{border-bottom:none}
  .who{font-family:"Press Start 2P",monospace;font-size:9px;letter-spacing:1px;margin-bottom:4px}
  .u .who{color:var(--mustard)} .a .who{color:var(--ok)}
  .u .body{background:rgba(217,168,108,.12);border-left:3px solid var(--mustard);padding:6px 9px;
    white-space:pre-wrap;overflow-wrap:anywhere}
  .a .body{padding:2px 0}

  /* ---- collapsible blocks ---- */
  details.blk{border:1px solid var(--dim);background:rgba(32,27,24,.75);margin:6px 0}
  details.blk>summary{list-style:none;cursor:pointer;padding:6px 9px;
    font-family:"Press Start 2P",monospace;font-size:9px;letter-spacing:1px;color:var(--dim);
    display:flex;align-items:center;gap:8px;user-select:none}
  details.blk>summary::-webkit-details-marker{display:none}
  details.blk>summary:hover{color:var(--mustard);background:rgba(217,168,108,.08)}
  details.blk .caret{display:inline-block;transition:transform .12s;transform:rotate(0deg)}
  details.blk[open] .caret{transform:rotate(90deg)}
  details.blk .cnt{color:var(--dim);font-family:"VT323",monospace;font-size:.9rem;margin-left:auto;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:55%}
  details.blk .inner{padding:7px 10px;border-top:1px dashed rgba(201,185,148,.22);
    white-space:pre-wrap;overflow-wrap:anywhere;font-size:1rem}
  .argp{font-family:"VT323",monospace;font-size:.9rem;color:var(--mustard);opacity:.85;
    max-width:52%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;letter-spacing:0}
  .think .inner{color:var(--dim);font-style:italic}
  .toolblk .inner{color:var(--bone)}
  .toolblk .args{color:var(--mustard);margin-bottom:6px;overflow-wrap:anywhere}
  .out{color:var(--ok)} .err{color:var(--brick)}
  .stream{white-space:pre-wrap;overflow-wrap:anywhere}
  .cursor::after{content:"_";color:var(--mustard);animation:blink 1s steps(2) infinite}
  @keyframes blink{50%{opacity:0}}
  .ctxbar{height:8px;background:var(--iron);border:1px solid var(--dim);margin-top:5px}
  .ctxbar i{display:block;height:100%;background:var(--ok);width:0%;transition:width .3s}
  .ctxbar i.hot{background:var(--mustard)} .ctxbar i.max{background:var(--brick)}

  /* ---- composer: always visible at the bottom ---- */
  .composer{flex:0 0 auto;border-top:3px solid var(--bone);padding:10px 12px;background:rgba(27,51,34,.72)}
  #task{min-height:52px;max-height:30vh;resize:none;overflow-y:auto;line-height:1.3}
  .composer .row{margin-top:7px}
  .hint{margin-left:auto;color:var(--dim);font-size:.85rem;opacity:.75}

  /* ---- tabs ---- */
  .tabs{display:flex;gap:5px;margin-left:6px}
  .tab{font-family:"Press Start 2P",monospace;font-size:9px;padding:6px 9px;letter-spacing:1px}
  .tab.on{background:var(--navy);border-color:var(--bone);color:var(--mustard)}
  .view{flex:1 1 auto;min-height:0;display:flex;flex-direction:column}
  .view[hidden]{display:none}

  /* ---- configuration ---- */
  .cfgwrap{flex:1 1 auto;min-height:0;overflow-y:auto;padding:14px 16px}
  .cfgcol{max-width:1240px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:13px;align-items:start}
  @media(max-width:980px){.cfgcol{grid-template-columns:1fr}}
  .cfgrow{display:flex;align-items:center;gap:9px;padding:4px 0;
    border-bottom:1px dotted rgba(201,185,148,.16)}
  .cfgrow .k{flex:1;min-width:0;color:var(--bone);font-size:1rem;
    overflow-wrap:break-word;word-break:normal}
  .cfgrow .k small{display:block;color:var(--dim);font-size:.83rem}
  .cfgrow input[type=number],.cfgrow input[type=text]{width:136px;padding:3px 6px;font-size:.95rem}
  .cfgrow input[type=checkbox]{width:18px;height:18px;accent-color:#7cb342}
  .cfgrow input[type=text]{flex:0 0 220px;width:220px}
  .good li{margin:5px 0;color:var(--dim)}
  .good b{color:var(--mustard)}
  .cfgbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
  #cfgsaved{color:var(--ok);font-size:.9rem}

  /* ---- narrow: stack, and let the page scroll (no nested traps) ---- */
  @media(max-width:880px){
    header{padding:7px 10px;gap:6px}
    .sub{display:none}
    h1{font-size:13px}
    .main{display:flex;flex-direction:column;flex:1 1 auto;min-height:0;padding:8px;gap:8px}
    .side{flex:0 0 auto;max-height:34vh;min-height:0;overflow-y:auto}
    .panel.sessions{flex:0 0 auto;min-height:0}
    #sesslist{max-height:none;min-height:0;overflow:visible}
    .chat{flex:1 1 auto;min-height:0}
    .composer{padding:8px}
    #task{min-height:56px;max-height:22vh}
    .hint{display:none}
  }
  @media(max-width:600px){
    .lbl{display:none}
    /* a phone cannot show a label and its input side by side: stack them */
    .cfgrow{display:block;padding:7px 0}
    .cfgrow .k{font-size:1rem;margin-bottom:3px}
    .cfgrow input[type=text]{width:100%;flex:1 1 auto;max-width:100%}
    .cfgrow input[type=number]{width:100%;max-width:180px}
    .cfgrow input[type=checkbox]{margin-top:4px}
    button.sm{padding:4px 5px;font-size:7px}
    #mpick{max-width:168px}
    .pill{padding:4px 5px;font-size:8px}
    .panel{padding:9px}
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="diamond"></div>
    <div>
      <h1>MINI AGENT</h1>
      <div class="sub"><span id="submodel">qwen3.5</span><span id="submeta"> &middot; local llama-server</span></div>
    </div>
  </div>
  <div class="tabs">
    <button class="tab on" id="tab-chat-btn" data-view="chat">CHAT</button>
    <button class="tab" id="tab-config-btn" data-view="config">CONFIG</button>
  </div>
  <div class="hgroup">
    <span id="srv" class="pill wait">MODEL ?</span>
    <span id="busy" class="pill">IDLE</span>
    <button class="sm" id="bcheck" title="Probe the model for the @@@ degeneration">CHECK</button>
    <button class="sm" id="brestart" title="Restart the model server (fresh runner = clean)">RESTART</button>
  </div>
  <div class="sep"></div>
  <div class="hgroup">
    <span class="meta lbl" title="Which model the runner loads. One at a time: the card cannot hold both.">model</span>
    <select id="mpick" style="width:auto;font-size:.95rem;padding:4px 6px"
      title="Switch model. APPLY restarts the runner (4B ~20 s, 9B ~60 s: 5.2 GiB of weights)."></select>
    <span class="meta lbl" title="Thinking tokens the model may spend before answering. Server-side flag: APPLY saves it and restarts the model.">budget</span>
    <input id="rbudget" type="number" min="0" max="4096" step="32" value="384"
      style="width:5.4em;padding:4px 6px;font-size:.95rem">
    <button class="sm" id="bapply" title="Save the reasoning budget and restart the model">APPLY</button>
    <span id="mstat" class="meta"></span>
  </div>
</header>

<section id="view-chat" class="view">
<div class="main">
  <aside class="side">
    <section class="panel sessions">
      <label for="sid">SESSION</label>
      <input id="sid" placeholder="(click a session, or NEW)" />
      <div class="row">
        <button class="sm" id="bnew">NEW</button>
        <button class="sm" id="bren">RENAME</button>
        <button class="sm" id="breset">RESET</button>
        <button class="sm" id="bcompact">COMPACT</button>
      </div>
      <div id="sesslist"></div>
      <div class="meta" id="sessinfo" style="margin-top:7px"></div>
    </section>

    <section class="panel">
      <label for="cwd">WORKING DIRECTORY</label>
      <input id="cwd" placeholder="C:\path\to\work" />
      <div id="dirs"></div>
    </section>

    <section class="panel two">
      <div>
        <label for="steps">MAX STEPS</label>
        <input id="steps" type="number" min="1" max="40" value="12" />
      </div>
      <div>
        <label title="The model sees the whole session each turn. When the bar gets hot, hit COMPACT or RESET.">CONTEXT</label>
        <div class="meta" id="ctxlabel">&mdash;</div>
        <div class="ctxbar"><i id="ctxfill"></i></div>
      </div>
    </section>
  </aside>

  <main class="chat">
    <div id="log"></div>
    <div class="composer">
      <textarea id="task" placeholder="e.g. list the python files here and summarise main.py"></textarea>
      <div class="row">
        <button class="go" id="go">RUN</button>
        <button class="stop" id="stop" disabled>STOP</button>
        <span class="meta" id="elapsed"></span>
        <span class="meta" id="stepsdone"></span>
        <span class="hint">Ctrl+Enter = run</span>
      </div>
    </div>
  </main>
</div>
</section>

<section id="view-config" class="view" hidden>
  <div class="cfgwrap">
    <div class="cfgcol">
      <div>
        <div class="panel" id="cfgform">
          <label>CONFIGURATION &mdash; config.json</label>
          <div class="meta" id="cfgpath"></div>
          <div id="cfgrows"></div>
          <div class="cfgbar">
            <button class="go" id="cfgsave">SAVE</button>
            <button class="sm" id="cfgreload">RELOAD</button>
            <span id="cfgsaved"></span>
          </div>
          <div class="meta" style="margin-top:8px">
            agent.* is read when a task starts. ui.* needs a page reload. tools.* and
            compaction.* are read at the next task.
          </div>
        </div>

        <div class="panel" style="margin-top:12px">
          <label>TOOLS &amp; BUDGETS</label>
          <div id="cfgtools"></div>
        </div>
      </div>

      <div>

        <div class="panel" style="margin-top:12px">
          <label>SYSTEM PROMPT &mdash; editable</label>
          <div class="meta" id="syspath"></div>
          <textarea id="sysedit" spellcheck="false" style="min-height:210px;margin-top:7px"></textarea>
          <div class="cfgbar">
            <button class="go" id="syssave">SAVE PROMPT</button>
            <button class="sm" id="sysrevert" title="load the built-in text into the box (does not save)">RESET BOX</button>
            <span class="meta" id="syscount"></span>
            <span id="sysstate" class="meta"></span>
          </div>
          <div class="meta" style="margin-top:7px">
            Sent as <b>messages[0]</b> on every turn &mdash; one copy per request, not
            appended per step. Applies to the next task.
          </div>
        </div>

        <div class="panel" style="margin-top:12px">
          <label>MEMORY &mdash; MEMORY.md / USER.md</label>
          <div class="meta" id="mempath"></div>
          <div class="cfgrow" style="margin-top:7px">
            <div class="k">MEMORY.md<small id="memcount">#</small></div>
            <span class="meta" id="memstate"></span>
          </div>
          <textarea id="memedit" spellcheck="false" style="min-height:150px"></textarea>
          <div class="cfgbar"><button class="sm" id="memsave">SAVE MEMORY.md</button></div>

          <div class="cfgrow" style="margin-top:9px">
            <div class="k">USER.md<small id="usrcount">#</small></div>
            <span class="meta" id="usrstate"></span>
          </div>
          <textarea id="usredit" spellcheck="false" style="min-height:120px"></textarea>
          <div class="cfgbar"><button class="sm" id="usrsave">SAVE USER.md</button></div>

          <div class="meta" style="margin-top:8px">
            These are Hermes's live memory files. A save writes them for real (with a
            <code>.bak</code>) and the agent only reads them when
            <b>prompt.inject_memory</b> is on.
          </div>
        </div>

        <div class="panel">
          <label>GOOD PRACTICES (what this build enforces)</label>
          <ol class="good">
            <li><b>Find before you read.</b> grep returns <code>path:line: text</code>;
                list returns names and sizes. Both are a fraction of a file read.</li>
            <li><b>Outline before you read.</b> <code>read mode=outline</code> gives sections,
                ids, functions and classes with line numbers (~5x cheaper: 1.4 KB vs 5 KB on
                a 431-line page).</li>
            <li><b>Read a range, not the file.</b> <code>read start=70 end=74</code> with line
                numbers; the window is capped instead of silently truncated.</li>
            <li><b>Never read the same file twice.</b> An unchanged file answers in one line and
                points at ranges instead.</li>
            <li><b>Clipping tells the truth.</b> The marker names how much was dropped, cuts on
                line boundaries and keeps the tail (logs and errors live there).</li>
            <li><b>Cheap compaction first.</b> Old tool payloads are trimmed before any model
                call is spent on a summary, and only above the trigger percentage.</li>
            <li><b>One writer per session.</b> A run holds a lock, so COMPACT/RESET refuse
                instead of racing the run and losing the summary.</li>
            <li><b>Timeouts kill the tree.</b> A command that spawns a server cannot survive its
                own timeout and hold the port.</li>
          </ol>
        </div>

      </div>
    </div>
  </div>
</section>

<script>
const $ = (id) => document.getElementById(id);
let ctrl = null, t0 = 0, tick = null, cur = null, running = false;

let CTX = 32768, DEFAULT_STEPS = 12;
function ctxNow(){ return CTX || 32768; }
function setMeter(tok, ctx, exact){
  tok = Math.max(0, Math.round(tok || 0)); ctx = ctx || ctxNow();
  const pct = Math.max(0, Math.min(100, (tok / ctx) * 100));
  const f = $('ctxfill'); f.style.width = pct + '%';
  f.className = pct > 85 ? 'max' : (pct > 55 ? 'hot' : '');
  $('ctxlabel').textContent = (exact ? '' : '~') + tok + ' / ' + ctx +
    ' (' + pct.toFixed(0) + '%)' + (exact ? ' exact' : ' est');
}
function esc(s){return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function nearBottom(){ const l=$('log'); return l.scrollHeight - l.scrollTop - l.clientHeight < 90; }
function scroll(){ const l=$('log'); if (nearBottom()) l.scrollTop = l.scrollHeight; }

/* ---- session helpers ---- */
async function refreshSessions(){
  const r = await fetch('/api/sessions', {cache:'no-store'});
  const d = await r.json();
  const list = $('sesslist'); list.innerHTML='';
  (d.sessions||[]).forEach(s=>{
    const b=document.createElement('button');
    if (s.id === $('sid').value) b.className='sel';
    if (s.id === d.running) b.className = (b.className ? b.className + ' ' : '') + 'running';
    const label = s.id.length > 26 ? s.id.slice(0,25) + '\u2026' : s.id;
    b.title = s.id;
    b.innerHTML = esc(label) + '<span class="t">' + esc((s.title||'').slice(0,40)) +
                   ' · ' + s.messages + ' msg' + (s.empty ? ' · empty' : '') + '</span>';
    b.onclick = ()=>{ $('sid').value=s.id; if(s.cwd) $('cwd').value=s.cwd;
                      refreshSessions(); loadTranscript(s.id); };
    list.appendChild(b);
  });
  const shown = $('sid').value.trim();
  if (d.current && (d.current.id === shown || (!shown && d.busy))){
    // only move the meter for the session the user is actually looking at
    $('sessinfo').textContent = d.current.messages + ' messages · ' +
                                (d.current.exact ? '' : '~') + d.current.tokens +
                                ' tokens · ' + (d.current.compacted ? 'compacted' : 'raw');
    setMeter(d.current.tokens, d.current.ctx_size, d.current.exact);
  }
}

async function newSession(){
  const r = await fetch('/api/session/new', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({cwd: $('cwd').value.trim()})
  });
  const d = await r.json();
  if (!d.ok){ alert(d.error || 'could not create the session'); return; }
  $('sid').value = d.id;
  $('log').innerHTML = '';
  await refreshSessions();
  setMeter(0, ctxNow(), false);
  $('sessinfo').textContent = '0 messages · new session';
  $('task').focus();
}

async function renameSession(){
  const id = $('sid').value.trim();
  if (!id){ alert('select a session first'); return; }
  const to = (prompt('New name for "' + id + '" (letters, digits, . - _)', id) || '').trim();
  if (!to || to === id) return;
  const r = await fetch('/api/session/rename', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, to})
  });
  const d = await r.json();
  if (!d.ok){ alert(d.error + (d.suggest ? ('\nTry: ' + d.suggest) : '')); return; }
  $('sid').value = d.id;
  await refreshSessions();
  addNote('renamed ' + id + ' -> ' + d.id);
}

async function sessAction(action){
  const id = $('sid').value.trim();
  if (!id){ alert('select a session first'); return; }
  const r = await fetch('/api/session/' + action, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, cwd:$('cwd').value.trim()})
  });
  const d = await r.json();
  if (d.error) { alert(d.error); return; }
  if (action === 'reset') { $('log').innerHTML=''; }
  if (action === 'compact') { addNote('COMPACT: ' + (d.summary||d.reason||'done')); }
  refreshSessions();
  if (action === 'compact') loadTranscript(id);
}

$('sid').addEventListener('change', async ()=>{
  const id = $('sid').value.trim();
  if (!id){ $('sessinfo').textContent = ''; setMeter(0, ctxNow(), false); return; }
  try{
    const d = await (await fetch('/api/context?id=' + encodeURIComponent(id))).json();
    if (!d.exists){
      // a name for a session that does not exist yet: RUN will create it
      $('sessinfo').textContent = 'new session - RUN will create it';
      setMeter(0, ctxNow(), false);
      return;
    }
    loadTranscript(id);
  }catch(e){}
});
$('bnew').onclick = ()=> newSession();
$('bren').onclick = ()=> renameSession();
$('breset').onclick = ()=> sessAction('reset');
$('bcompact').onclick = ()=> sessAction('compact');

/* ---- composer: grows with the text, Ctrl+Enter runs ---- */
const taskEl = $('task');
function grow(){ taskEl.style.height='auto';
  taskEl.style.height = Math.min(taskEl.scrollHeight, Math.round(innerHeight*0.30)) + 'px'; }
taskEl.addEventListener('input', grow); grow();
taskEl.addEventListener('keydown', e=>{
  if (e.key==='Enter' && (e.ctrlKey || e.metaKey)){ e.preventDefault();
    if (! $('go').disabled) $('go').click(); }
});

/* ---- server status ---- */
let pollFails = 0;
async function pollServer(){
  try{
    const s = await (await fetch('/api/status', {cache:'no-store'})).json();
    pollFails = 0;
    const p = $('srv');
    const mm = !!s.model_mismatch;   // port up but the wrong weights are loaded
    p.textContent = s.model_up ? (mm ? 'WRONG MODEL' : 'MODEL UP') : 'MODEL DOWN';
    p.className = 'pill ' + (s.model_up && !mm ? 'on' : 'off');
    p.title = s.load_error || (mm ? ('loaded "' + (s.loaded_model||'?') + '" / profile wants "' +
                                   (s.model_expected||'?') + '"') : '');
    const warn = !s.model_up ? (s.load_error || '')
               : (mm ? ('the loaded model is "' + (s.loaded_model||'?') + '" but the profile wants "' +
                        (s.model_expected||'?') + '" - press RESTART') : '');
    if (warn){
      $('mstat').textContent = warn; $('mstat').dataset.le = '1';
      $('mstat').style.color = '#e0564a';
    } else if ($('mstat').dataset.le){
      $('mstat').textContent = ''; $('mstat').style.color = ''; delete $('mstat').dataset.le;
    }
    $('busy').textContent = s.busy ? 'BUSY' : 'IDLE';
    $('busy').className = 'pill' + (s.busy ? ' wait' : '');
    /* STOP must work for a run this page did not start (after an F5, or from a
       second tab): drive the buttons from the SERVER's busy flag, not from the
       local `running` variable. Before this a reloaded page showed the run
       streaming with STOP greyed out, so there was no way to stop it. */
    $('stop').disabled = !s.busy;
    $('go').disabled = !!s.busy;
    if (s.reasoning_budget !== undefined && !$('rbudget').dataset.touched)
      $('rbudget').value = s.reasoning_budget;
    const sel = $('mpick');
    if (s.profiles && sel.options.length !== s.profiles.length){
      sel.innerHTML = '';
      for (const p of s.profiles){
        const o = document.createElement('option');
        o.value = p.id; o.textContent = p.label;
        sel.appendChild(o);
      }
    }
    if (s.profile && !sel.dataset.touched) sel.value = s.profile;
    if (s.profile_label && $('submodel')) $('submodel').textContent = s.profile_label.split(' - ')[0].toLowerCase();
    if (s.context && s.context !== CTX){ CTX = s.context; }
    // while a run is going, refresh the meter from the session file (the agent
    // saves after every step) so it never freezes mid-run
    if (running && $('sid').value.trim()){
      const c = await (await fetch('/api/context?id=' + encodeURIComponent($('sid').value.trim()),
                                   {cache:'no-store'})).json();
      if (c.exists) setMeter(c.tokens, c.ctx_size, c.exact);
    }
    // the session list must show WHICH session is running (the marker needs a
    // fresh /api/sessions response, which the boot/click paths do not provide)
    if (s.busy) refreshSessions();
  }catch(e){
    if (++pollFails >= 2){
      const p = $('srv'); p.textContent = 'PAGE OFFLINE'; p.className = 'pill off';
      p.title = 'the web server is not answering';
    }
  }
}
setInterval(pollServer, 3000); pollServer();

/* ---- dirs ---- */
fetch('/api/dirs').then(r=>r.json()).then(d=>{
  const box=$('dirs'); box.innerHTML='';
  d.dirs.forEach(x=>{
    const b=document.createElement('button');
    b.textContent=x; b.title=x;
    b.onclick=()=>{ $('cwd').value=x; };
    box.appendChild(b);
  });
});

/* ---- transcript builders ---- */
function addUser(text){
  const d=document.createElement('div'); d.className='msg u';
  d.innerHTML='<div class="who">TASK</div><div class="body">'+esc(text)+'</div>';
  $('log').appendChild(d); scroll();
}
function addNote(text, cls){
  const d=document.createElement('div'); d.className='msg'+(cls?' '+cls:'');
  d.innerHTML='<div class="who '+(cls||'meta')+'">'+esc(text)+'</div>';
  $('log').appendChild(d); scroll();
}
function addThinkBlock(){
  const d=document.createElement('details'); d.className='blk think';
  d.innerHTML='<summary><span class="caret">▸</span> THINKING<span class="cnt">0 chars</span></summary>'+
              '<div class="inner stream"></div>';
  $('log').appendChild(d); scroll();
  return {root:d, inner:d.querySelector('.inner'), cnt:d.querySelector('.cnt')};
}
function addToolBlock(name, args){
  const d=document.createElement('details'); d.className='blk toolblk';
  const a=args||{}, hint=String(a.path||a.cmd||a.pattern||a.query||'').replace(/\s+/g,' ').trim();
  d.innerHTML='<summary><span class="caret">▸</span> TOOL: '+esc(name)+
              (hint?'<span class="argp" title="'+esc(hint)+'">'+esc(hint.slice(0,80))+'</span>':'')+
              '<span class="cnt">running…</span></summary>'+
              '<div class="inner"><div class="args">'+esc(JSON.stringify(args||{}))+'</div>'+
              '<div class="out"></div></div>';
  $('log').appendChild(d); scroll();
  return {root:d, cnt:d.querySelector('.cnt'), out:d.querySelector('.out')};
}
function addSayBlock(){
  const d=document.createElement('div'); d.className='msg a';
  d.innerHTML='<div class="who">ASSISTANT</div><div class="body stream"></div>';
  $('log').appendChild(d); scroll();
  return d.querySelector('.body');
}

/* ---- model control: check + restart (manual, no auto-restart) ---- */
async function modelAction(action){
  if (action === 'restart'){
    if (!confirm('Restart the model server?\n\nThis kills the current run (the session is saved) and\nreloads the model into VRAM.')) return;
    $('brestart').disabled = true;
    $('mstat').textContent = 'restarting...';
    try{
      const r = await fetch('/api/model/restart', {method:'POST'});
      const d = await r.json();
      const det = ((d.steps||[]).find(s=>s.step==='detail')||{}).error;
      $('mstat').textContent = d.model_up
        ? ('restarted in ' + d.seconds + 's')
        : ('RESTART FAILED: ' + (det || (d.steps||[]).map(s=>s.step+':'+(s.error||s.rc)).join(' ')));
    }catch(e){ $('mstat').textContent = 'restart failed: ' + e.message; }
    $('brestart').disabled = false;
    pollServer();
    return;
  }
  // check
  $('bcheck').disabled = true;
  $('mstat').textContent = 'checking...';
  try{
    const r = await fetch('/api/model/check', {method:'POST'});
    const d = await r.json();
    if (d.ok) $('mstat').textContent = 'model is CLEAN (' + d.seconds + 's)';
    else if (d.state === 'poisoned') $('mstat').textContent = 'POISONED - hit RESTART';
    else $('mstat').textContent = 'model unreachable: ' + d.detail;
  }catch(e){ $('mstat').textContent = 'check failed: ' + e.message; }
  $('bcheck').disabled = false;
}
$('bcheck').onclick = ()=> modelAction('check');
$('brestart').onclick = ()=> modelAction('restart');
// APPLY = save the budget AND the model profile, then restart ONCE.
// Both are start-time facts (--reasoning-budget, --model/--ctx-size), so a
// single restart applies them; two restarts would just cost a minute of VRAM churn.
$('rbudget').addEventListener('input', ()=>{ $('rbudget').dataset.touched = '1'; });
$('mpick').addEventListener('change', ()=>{ $('mpick').dataset.touched = '1'; });
$('bapply').onclick = async ()=>{
  $('bapply').disabled = true;
  $('mstat').textContent = 'saving...';
  try{
    const val = parseInt($('rbudget').value || '384', 10);
    const r = await fetch('/api/tuning', {method:'POST', headers:{'Content-Type':'application/json'},
                                          body: JSON.stringify({budget: val})});
    const d = await r.json();
    if (d.error){ $('mstat').textContent = 'budget: ' + d.error; $('bapply').disabled = false; return; }
    const want = $('mpick').value;
    const r2 = await fetch('/api/profile', {method:'POST', headers:{'Content-Type':'application/json'},
                                            body: JSON.stringify({id: want})});
    const d2 = await r2.json();
    if (d2.error){ $('mstat').textContent = 'model: ' + d2.error; $('bapply').disabled = false; return; }
    const lbl = ($('mpick').selectedOptions[0] || {}).textContent || want;
    $('mstat').textContent = 'budget ' + d.budget + ' - ' + lbl + ' - restarting...';
    $('rbudget').dataset.touched = ''; $('mpick').dataset.touched = '';
    await modelAction('restart');
  }catch(e){ $('mstat').textContent = 'apply failed: ' + e.message; }
  $('bapply').disabled = false;
};

/* ---- load a saved session's transcript ---- */
async function loadTranscript(sid){
  if (!sid) return;
  let d;
  try{
    const r = await fetch('/api/session?id=' + encodeURIComponent(sid), {cache:'no-store'});
    d = await r.json();
  }catch(e){ addNote('could not load the session: ' + e.message); return; }
  if (d.error){ addNote('could not load "'+sid+'": ' + d.error); return; }

  $('log').innerHTML = '';
  addNote('SESSION ' + d.id + ' · ' + (d.messages||[]).length + ' messages · ' +
          (d.exact ? '' : '~') + d.tokens + ' tokens' + (d.compacted ? ' · compacted' : ''));

  // point the context meter at the session the user is looking at
  setMeter(d.tokens, d.ctx_size, d.exact);
  $('sessinfo').textContent = (d.messages||[]).length + ' messages · ' +
                              (d.exact ? '' : '~') + d.tokens +
                              ' tokens · ' + (d.compacted ? 'compacted' : 'raw');

  const tr = d.transcript || [];
  if (tr.length){
    // The rich transcript is the source of truth: it keeps the reasoning, the
    // tool call ARGS and the tool OUTPUT even after a compaction.
    const pending = [];   // tool blocks waiting for their result
    for (const e of tr){
      if (e.kind === 'user'){
        addUser(e.text || '');
      } else if (e.kind === 'assistant'){
        if (e.reasoning){
          const b = addThinkBlock();
          b.inner.textContent = e.reasoning;
          b.cnt.textContent = e.reasoning.length + ' chars';
        }
        for (const tc of (e.tool_calls || [])){
          const b = addToolBlock(tc.name, safeArgs(tc.arguments));
          b.cnt.textContent = 'running...';
          pending.push(b);
        }
        if (e.content){
          const body = addSayBlock();
          body.textContent = e.content;
        }
      } else if (e.kind === 'tool'){
        // fill the matching pending block (call + result in ONE block)
        let b = pending.shift();
        if (!b){
          b = addToolBlock(e.name, e.args || {});
        } else if (e.args && Object.keys(e.args).length){
          b.root.querySelector('.args').textContent = JSON.stringify(e.args);
        }
        b.out.textContent = e.output || '';
        b.cnt.textContent = (e.output || '').length + ' chars';
      }
    }
    scroll();
    return;
  }

  const msgs = d.messages || [];
  for (let i=0; i<msgs.length; i++){
    const m = msgs[i];
    if (m.role === 'system') continue;
    if (m.role === 'user'){
      let txt = m.content || '';
      if (txt.indexOf('[COMPACTED SUMMARY') === 0){
        const b = addThinkBlock();
        b.inner.textContent = txt;
        b.cnt.textContent = txt.length + ' chars';
        b.root.querySelector('summary').childNodes[1].nodeValue = ' COMPACTED SUMMARY ';
        continue;
      }
      txt = txt.replace(/^Working directory:[\s\S]*?\n\nTask:\s*/, '');
      addUser(txt);
    }
    else if (m.role === 'assistant'){
      const tcs = m.tool_calls || [];
      // the reasoning is persisted with the message -> show it again
      if (m.reasoning){
        const b = addThinkBlock();
        b.inner.textContent = m.reasoning;
        b.cnt.textContent = m.reasoning.length + ' chars';
      }
      if (tcs.length){
        addNote('calling: ' + tcs.map(t=>(t.function||{}).name).join(', '));
        const next = msgs[i+1];
        if (next && next.role === 'tool'){
          const b = addToolBlock(tcs[0].function.name, safeArgs(tcs[0].function.arguments));
          b.out.textContent = stripToolHeader(next.content || '');
          b.cnt.textContent = b.out.textContent.length + ' chars';
        }
      }
      if (m.content){
        const body = addSayBlock();
        body.textContent = m.content;
      }
    }
    else if (m.role === 'tool'){
      const prev = msgs[i-1] || {};
      if (prev.role === 'assistant' && (prev.tool_calls||[]).length) continue;
      // orphaned tool result (its assistant message was compacted away):
      // the tool name is in the "[name]" header
      const hdr = String(m.content || '').match(/^\[([^\]]*)\]/);
      const b = addToolBlock(hdr ? hdr[1] : 'tool', {});
      b.out.textContent = stripToolHeader(m.content || '');
      b.cnt.textContent = b.out.textContent.length + ' chars';
    }
  }
  scroll();
}

function safeArgs(a){
  if (!a) return {};
  try{ return JSON.parse(a); }catch(e){ return {raw: String(a).slice(0,200)}; }
}
function stripToolHeader(c){
  const m = String(c).match(/^\[[^\]]*\]\n?/);
  return m ? String(c).slice(m[0].length) : String(c);
}

/* ---- run ---- */
$('go').onclick = async () => {
  const task = $('task').value.trim();
  if (!task) return;
  addUser(task);
  $('task').value = ''; grow();          // the box is cleared when the task leaves it
  t0 = Date.now();
  tick = setInterval(()=>{ $('elapsed').textContent = ((Date.now()-t0)/1000).toFixed(0)+'s'; }, 500);
  $('go').disabled = true; $('stop').disabled = false;
  ctrl = new AbortController();
  running = true;
  cur = {think:null, say:null, tool:null};
  $('stepsdone').textContent = '';
  try{
    const res = await fetch('/api/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        task, cwd: $('cwd').value.trim(),
        session: $('sid').value.trim(),
        steps: parseInt($('steps').value,10)||DEFAULT_STEPS
      }),
      signal: ctrl.signal
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder(); let buf='';
    while(true){
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      let i;
      while((i = buf.indexOf('\n')) >= 0){
        const line = buf.slice(0,i); buf = buf.slice(i+1);
        if (!line.trim()) continue;
        let ev; try{ ev = JSON.parse(line); }catch{ continue; }
        handle(ev);
      }
    }
  }catch(e){
    if (e.name !== 'AbortError'){
      addNote('ERROR: ' + e.message, 'err');
      addNote('the task above was NOT saved - the run never started');
    }
  }finally{
    clearInterval(tick);
    running = false;
    $('go').disabled = false; $('stop').disabled = true;
    $('elapsed').textContent = ((Date.now()-t0)/1000).toFixed(0)+'s';
    pollServer(); refreshSessions();
  }
};
// ---- a refresh or a closed tab does NOT stop the agent ----
// It used to: pagehide/beforeunload posted /api/cancel, so an F5 threw away a
// running task. The server now keeps the run going (and keeps draining its
// output), the page re-attaches on load and shows BUSY, and only the STOP
// button cancels. The session is saved after every step either way.

$('stop').onclick = async ()=>{
  if (ctrl) ctrl.abort();
  addNote('STOPPING the agent...');
  try{
    const r = await fetch('/api/cancel', {method:'POST'});
    const d = await r.json();
    addNote(d.ok ? ('STOPPED - ' + (d.detail||'')) : ('could not stop: ' + (d.detail||d.error||'')));
  }catch(e){ addNote('cancel failed: ' + e.message); }
};

function handle(ev){
  if (ev.type === 'turn'){
    /* One block per MODEL TURN. Without this the reasoning of step 2 was appended
       into the step-1 THINKING box (and every say into one bubble); a reload looked
       right only because it rebuilds from the stored transcript. */
    cur = {think:null, say:null, tool:null};
    cur.step = ev.step;
    return;
  }
  if (ev.type === 'delta'){
    if (ev.kind === 'reasoning'){
      if (!cur.think) cur.think = addThinkBlock();
      cur.think.inner.textContent += ev.text;
      cur.think.cnt.textContent = cur.think.inner.textContent.length + ' chars';
      scroll();
    } else if (ev.kind === 'content'){
      if (!cur.say) cur.say = addSayBlock();
      cur.say.textContent += ev.text;
      scroll();
    } else if (ev.kind === 'toolargs'){
      /* tool args stream into the block that will be created on the tool event */
    }
    return;
  }
  if (ev.type === 'start'){
    // the agent creates a session when RUN is pressed with none selected:
    // adopt it, otherwise every run would create yet another one
    if (ev.session && !$('sid').value.trim()){
      $('sid').value = ev.session;
      refreshSessions();
    }
    addNote('session: ' + (ev.session||'(one-off)') +
            (ev.resumed ? ' · resumed ' + ev.resumed + ' msgs' : ' · new') +
            ' · ~' + ev.tokens_before + ' tok');
  }
  else if (ev.type === 'assistant'){
    $('stepsdone').textContent = 'step ' + ev.step + ' · ' + ev.seconds + 's';
    if (ev.tool_calls && ev.tool_calls.length) addNote('calling: ' + ev.tool_calls.join(', '));
  }
  else if (ev.type === 'tool'){
    const b = addToolBlock(ev.name, ev.args);
    b.out.textContent = ev.output;
    b.cnt.textContent = ev.output.length + ' chars';
  }
  else if (ev.type === 'done'){
    if (ev.failed) addNote('FAILED — ' + ev.reason, 'err');
    else addNote('DONE — ' + ev.reason);
  }
  else if (ev.type === 'error'){
    addNote('ERROR: ' + ev.message, 'err');
    if (ev.hint) addNote(ev.hint);
  }
  else if (ev.type === 'note'){ addNote(ev.message); }
  else if (ev.type === 'tokens'){
    // the server's own prompt_tokens for the call that just finished
    setMeter(ev.prompt, ev.ctx, true);
    if (ev.estimate && ev.prompt)
      $('ctxlabel').title = 'exact ' + ev.prompt + ' · estimate ' + ev.estimate +
                            ' (' + (100*(ev.estimate-ev.prompt)/ev.prompt).toFixed(0) + '% off)';
  }
  else if (ev.type === 'saved'){
    setMeter(ev.tokens, ev.ctx, ev.exact);
    addNote('saved · ' + ev.messages + ' msgs · ' + (ev.exact ? '' : '~') + ev.tokens +
            ' tok (' + (ev.usage*100).toFixed(0) + '% of ctx)');
  }
}

refreshSessions();
loadConfig();          // title + tool count come from config.json

/* ------------------------------------------------------------------ tabs */
const VIEWS = ['chat', 'config'];
function showView(v){
  VIEWS.forEach(x => {
    $('view-' + x).hidden = (x !== v);
    $('tab-' + x + '-btn').classList.toggle('on', x === v);
  });
  if (v === 'config') { loadConfig(); loadPrompt(); }
}
$('tab-chat-btn').onclick = () => showView('chat');
$('tab-config-btn').onclick = () => showView('config');

/* ---------------------------------------------------------- configuration */
const HINTS = {
  'agent.server': 'llama-server base URL',
  'agent.model': 'alias the runner must serve',
  'agent.ctx_size': 'context window of the loaded model (synced when you switch profiles)',
  'agent.max_steps': 'tool-call turns per task (UI clamps 1..40)',
  'agent.max_tokens': 'reply ceiling; clamped per step by the room left',
  'agent.temperature': 'sampling temperature',
  'agent.overhead_tokens': 'system prompt + tool schemas, counted in the ctx estimate',
  'agent.retry_cut_tool_call': 'one retry when the tool-call JSON is cut off',
  'tools.max_result_chars': 'cap on ONE tool result',
  'tools.bash_tail_ratio': 'share of the clip kept from the END (logs)',
  'tools.step_cap_multiplier': 'combined cap for all results of one step',
  'tools.read.max_lines': 'hard cap for a start/end window',
  'tools.read.outline_over_lines': 'above this many lines, read returns the MAP first',
  'tools.read.head_lines': 'lines shown after the map',
  'tools.grep.max_matches': 'stop after N match lines',
  'tools.grep.context': 'lines before/after each hit',
  'tools.bash.timeout_s': 'seconds before the process tree is killed',
  'tools.cache.reads': 'answer "unchanged" instead of re-reading a file',
  'compaction.auto': 'compact during a run when the context gets hot',
  'compaction.trigger_pct': 'fraction of the window that triggers it',
  'compaction.keep_last_tokens': 'tail budget kept verbatim',
  'compaction.trim_tool_results_first': 'clear old tool payloads before any model call',
  'ui.port': 'page port (reload after changing)',
  'ui.sessions_dir': 'where sessions are stored',
};
let CFG_JSON = null;
function flatten(o, prefix){
  const out = [];
  for (const [k, v] of Object.entries(o)){
    if (k.startsWith('_')) continue;
    const key = prefix ? prefix + '.' + k : k;
    if (v && typeof v === 'object' && !Array.isArray(v)) out.push(...flatten(v, key));
    else out.push({key, value: v, arr: Array.isArray(v)});
  }
  return out;
}
function setDeep(root, key, val){
  const parts = key.split('.'); let o = root;
  for (let i = 0; i < parts.length - 1; i++) o = o[parts[i]] = o[parts[i]] || {};
  o[parts[parts.length - 1]] = val;
}
async function loadConfig(){
  const r = await fetch('/api/config', {cache: 'no-store'});
  const d = await r.json();
  CFG_JSON = d.config;
  $('cfgpath').textContent = d.path;
  document.querySelector('h1').textContent = (d.config.ui && d.config.ui.title) || 'MINI AGENT v2';
  document.title = (d.config.ui && d.config.ui.title) || 'MINI AGENT v2';
  const sm = $('submeta');
  if (sm) sm.textContent = ' \u00b7 local llama-server \u00b7 ' + (d.tools || []).length + ' tools';
  const box = $('cfgrows'); box.innerHTML = '';
  let section = null;
  flatten(d.config, '').forEach(item => {
    const sec = item.key.split('.')[0];
    if (sec !== section){
      section = sec;
      const h = document.createElement('label');
      h.textContent = sec.toUpperCase();
      h.style.marginTop = '10px';
      box.appendChild(h);
    }
    const row = document.createElement('div'); row.className = 'cfgrow';
    const lab = document.createElement('div'); lab.className = 'k';
    lab.innerHTML = esc(item.key) + (HINTS[item.key] ? '<small>' + esc(HINTS[item.key]) + '</small>' : '');
    row.appendChild(lab);
    let inp;
    if (typeof item.value === 'boolean'){
      inp = document.createElement('input'); inp.type = 'checkbox'; inp.checked = item.value;
    } else if (typeof item.value === 'number'){
      inp = document.createElement('input'); inp.type = 'number'; inp.step = 'any'; inp.value = item.value;
    } else {
      inp = document.createElement('input'); inp.type = 'text';
      inp.value = item.arr ? JSON.stringify(item.value) : (item.value == null ? '' : item.value);
    }
    inp.dataset.key = item.key;
    inp.dataset.kind = typeof item.value === 'boolean' ? 'bool'
                      : typeof item.value === 'number' ? 'num'
                      : item.arr ? 'arr' : 'str';
    row.appendChild(inp);
    box.appendChild(row);
  });
  const tools = $('cfgtools'); tools.innerHTML = '';
  (d.tools || []).forEach(t => {
    const div = document.createElement('div'); div.className = 'cfgrow';
    div.innerHTML = '<div class="k">' + esc(t.name) + '<small>' + esc(t.desc) + '</small></div>' +
                    '<span class="meta">' + t.budget + ' ch</span>';
    tools.appendChild(div);
  });
}
async function loadPrompt(){
  try {
    const c = await (await fetch('/api/config')).json();
    if (c && c.config && c.config.agent){
      $('steps').value = c.config.agent.max_steps;
      DEFAULT_STEPS = c.config.agent.max_steps|0 || 12;
      if (!CTX) CTX = c.config.agent.ctx_size|0 || 32768;
    }
  } catch(e){}
  const r = await fetch('/api/prompt', {cache: 'no-store'});
  const d = await r.json();
  $('syspath').textContent = d.path + (d.inject_memory ? '  ·  memory injected' : '  ·  memory NOT injected');
  $('sysedit').value = d.system || '';
  $('syscount').textContent = (d.system || '').length + ' chars ~' + Math.round((d.system || '').length / 3.6) + ' tok';
  window.SYS_DEFAULT = d.default || '';
  $('mempath').textContent = d.memory.memory.path + '   |   ' + d.memory.user.path;
  $('memedit').value = d.memory.memory.text || '';
  $('usredit').value = d.memory.user.text || '';
  $('memcount').textContent = d.memory.memory.chars + ' chars (budget ' + d.memory.memory.budget + ')';
  $('usrcount').textContent = d.memory.user.chars + ' chars (budget ' + d.memory.user.budget + ')';
}
function syncCounts(){
  $('syscount').textContent = $('sysedit').value.length + ' chars ~' +
                              Math.round($('sysedit').value.length / 3.6) + ' tok';
  $('memcount').textContent = $('memedit').value.length + ' chars';
  $('usrcount').textContent = $('usredit').value.length + ' chars';
}
$('sysedit').addEventListener('input', syncCounts);
$('memedit').addEventListener('input', syncCounts);
$('usredit').addEventListener('input', syncCounts);

$('sysrevert').onclick = () => { $('sysedit').value = window.SYS_DEFAULT || ''; syncCounts();
  $('sysstate').textContent = 'box reset - press SAVE to apply'; };
$('syssave').onclick = async () => {
  const r = await fetch('/api/prompt', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({system: $('sysedit').value})});
  const d = await r.json();
  $('sysstate').textContent = d.ok ? 'saved · ' + d.chars + ' chars' : ('error: ' + (d.error || '?'));
  if (d.ok) loadConfig();
};
async function saveMem(which, box, state){
  const r = await fetch('/api/memory', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({which, text: box.value})});
  const d = await r.json();
  state.textContent = d.ok ? ('saved · ' + d.chars + ' chars') : ('error: ' + (d.error || '?'));
}
$('memsave').onclick = () => saveMem('memory', $('memedit'), $('memstate'));
$('usrsave').onclick = () => saveMem('user', $('usredit'), $('usrstate'));

$('cfgreload').onclick = () => loadConfig();
$('cfgsave').onclick = async () => {
  const out = JSON.parse(JSON.stringify(CFG_JSON));
  document.querySelectorAll('#cfgrows input').forEach(inp => {
    const k = inp.dataset.key, kind = inp.dataset.kind;
    let v;
    if (kind === 'bool') v = inp.checked;
    else if (kind === 'num') v = Number(inp.value);
    else if (kind === 'arr') { try { v = JSON.parse(inp.value); } catch(e){ v = inp.value.split(','); } }
    else v = inp.value;
    setDeep(out, k, v);
  });
  const r = await fetch('/api/config', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({config: out})
  });
  const d = await r.json();
  $('cfgsaved').textContent = d.ok ? 'saved' : ('error: ' + (d.error || '?'));
  if (d.ok) setTimeout(() => $('cfgsaved').textContent = '', 2500);
};
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return {}

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        if self.path.startswith("/api/status"):
            up = False
            try:
                urllib.request.urlopen("http://127.0.0.1:8099/v1/models", timeout=3)
                up = True
            except Exception:
                up = False
            cur = current_profile()
            served = loaded_alias() if up else ""
            want = PROFILES[cur]["alias"]
            self._json({"model_up": up, "busy": STATE["busy"], "task": STATE["task"],
                        "load_error": "" if up else _load_error(),
                          "loaded_model": served, "model_expected": want,
                          "model_mismatch": bool(up and served and served != want),
                          "reasoning_budget": tuning_budget(),
                          "profile": cur, "profile_label": PROFILES[cur]["label"],
                          "context": PROFILES[cur]["ctx"],
                          "profiles": [{"id": k, "label": v["label"], "ctx": v["ctx"]}
                                       for k, v in PROFILES.items()]})
        elif self.path.startswith("/api/prompt"):
            meta = agent_meta()
            pr = CFG["prompt"]
            try:
                system = meta["load_system"](CFG) if meta.get("load_system") else meta.get("system", "")
            except Exception as e:
                system = "error: %s" % e
            mem_txt = _safe_read(pr["memory_file"])
            usr_txt = _safe_read(pr["user_file"])
            self._json({
                "ok": True,
                "system": system,
                "default": meta.get("system") or "",
                "path": str(meta.get("prompt_file")(CFG)) if meta.get("prompt_file") else "",
                "inject_memory": bool(pr.get("inject_memory")),
                "memory": {
                    "memory": {"path": pr["memory_file"], "text": mem_txt,
                               "chars": len(mem_txt), "budget": pr.get("memory_budget", 2200)},
                    "user": {"path": pr["user_file"], "text": usr_txt,
                             "chars": len(usr_txt), "budget": pr.get("user_budget", 1375)},
                },
            })
        elif self.path.startswith("/api/config"):
            meta = agent_meta()
            self._json({
                "ok": True, "path": CONFIG_PATH, "config": CFG,
                "system": meta.get("system") or "",
                "tools": [{"name": t["function"]["name"],
                           "desc": t["function"]["description"][:140],
                           "budget": CFG["tools"]["max_result_chars"]}
                          for t in (meta.get("tools") or [])],
            })
        elif self.path.startswith("/api/dirs"):
            self._json({"dirs": agent_roots()})
        elif self.path.startswith("/api/session?"):
            q = urllib.parse.urlparse(self.path).query
            sid = urllib.parse.parse_qs(q).get("id", [""])[0].strip()
            if not sid:
                self._json({"error": "session id required"}, 400)
                return
            fp = session_file(sid)
            if not os.path.isfile(fp):
                self._json({"error": "no such session", "id": sid}, 404)
                return
            try:
                d = json.loads(_read_text_shared(fp))
            except Exception as e:
                self._json({"error": "this session file is not valid JSON (%s) - "
                                     "the agent will start a fresh one" % e,
                            "corrupt": True, "id": sid}, 500)
                return
            msgs = d.get("messages") or []
            tok, exact = session_tokens(d)
            ctxsz = PROFILES.get(current_profile(), {}).get("ctx", 32768)
            self._json({"id": d.get("id", sid), "title": d.get("title", ""),
                        "cwd": d.get("cwd", ""), "messages": msgs,
                        "transcript": d.get("transcript") or [],
                        "tokens": tok, "exact": exact, "ctx_size": ctxsz,
                        "last_usage": d.get("last_usage") or {},
                        "compacted": d.get("compacted", False)})
        elif self.path.startswith("/api/context"):
            q = urllib.parse.urlparse(self.path).query
            sid = urllib.parse.parse_qs(q).get("id", [""])[0].strip()
            fp = session_file(sid) if sid else ""
            if not sid or not os.path.isfile(fp):
                self._json({"exists": False, "id": sid})
                return
            try:
                d = json.loads(_read_text_shared(fp))
            except Exception as e:
                self._json({"exists": False, "id": sid, "error": str(e)})
                return
            tok, exact = session_tokens(d)
            ctxsz = PROFILES.get(current_profile(), {}).get("ctx", 32768)
            self._json({"exists": True, "id": d.get("id", sid),
                        "messages": len(d.get("messages") or []), "tokens": tok,
                        "exact": exact, "ctx_size": ctxsz,
                        "usage": min(1.0, tok / ctxsz)})
        elif self.path.startswith("/api/sessions"):
            cur = None
            with LOCK:
                sid = STATE.get("session") or ""
                busy = bool(STATE.get("busy"))
            if sid and os.path.isfile(session_file(sid)):
                try:
                    d = json.loads(_read_text_shared(session_file(sid)))
                    msgs = d.get("messages") or []
                    tok, exact = session_tokens(d)
                    ctxsz = PROFILES.get(current_profile(), {}).get("ctx", 32768)
                    cur = {"id": d.get("id"), "messages": len(msgs), "tokens": tok,
                           "exact": exact, "compacted": d.get("compacted", False),
                           "ctx_size": ctxsz, "usage": min(1.0, tok / ctxsz),
                           "running": busy and sid == STATE.get("session")}
                except Exception:
                    cur = None
            self._json({"sessions": read_sessions(), "current": cur,
                        "running": STATE.get("session") if busy else ""})
        elif self.path == "/" or self.path.startswith("/index"):
            page = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        else:
            self._json({"error": "not found"}, 404)


    # ------------------------------------------------------------------
    # model control (manual - the automatic restart is DISABLED on purpose)
    # ------------------------------------------------------------------
    def do_model_check(self):
        """Probe the model for the '@@@' degeneration. Never restarts anything.

        FIX 2026-10-21: the probe used to send max_tokens=120, which is TOO
        SHORT to reproduce the failure. The measured failures emitted the FULL
        max_tokens (4096) of '@' with finish_reason=length, while short
        generations stayed clean -- a 120-token probe therefore scores a
        POISONED runner as "clean" (a false negative). The reference notes
        say it outright: "A test capped at 200-400 tokens can miss it
        completely, which is why several 'fixes' looked successful."

        The probe now asks for a long answer (1200 tokens) and additionally
        asserts that output EXISTS -- a checker that only counts '@'/'0' runs
        scores "no output" as clean.
        """
        import re as _re
        body = {
            # follow the selected profile: llama-server rejects a model id it
            # does not serve, so a hardcoded alias would break CHECK on the 9B.
            "model": PROFILES[current_profile()]["alias"],
            "messages": [{"role": "user", "content":
                          "Think step by step and explain in detail how a thermistor "
                          "converts temperature into an electrical signal. At least "
                          "three paragraphs."}],
            "stream": False, "max_tokens": 1200,
            "temperature": 0.6, "presence_penalty": 0, "repeat_penalty": 1.1,
        }
        t0 = time.time()
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8099/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
        except Exception as e:
            self._json({"ok": False, "state": "unreachable",
                        "detail": f"{type(e).__name__}: {str(e)[:120]}"})
            return
        m = (d.get("choices") or [{}])[0].get("message") or {}
        content = m.get("content") or ""
        reasoning = m.get("reasoning_content") or ""
        blob = content + reasoning
        at = len(_re.findall(r"@{4,}", blob))
        zeros = len(_re.findall(r"0{4,}", blob))
        # any single character repeated 12+ times is also a degeneration
        degen = _re.findall(r"(.)\1{11,}", blob)
        if at or zeros or degen:
            self._json({"ok": False, "state": "poisoned",
                        "detail": f"{at} '@' run(s), {zeros} zero-run(s), "
                                  f"{len(degen)} repeated-char run(s)",
                        "seconds": round(time.time() - t0, 1)})
        elif not blob.strip():
            # empty output is NOT clean -- assert that output exists
            self._json({"ok": False, "state": "empty",
                        "detail": "the model returned no content AND no reasoning",
                        "seconds": round(time.time() - t0, 1)})
        else:
            self._json({"ok": True, "state": "clean",
                        "detail": (content or reasoning).strip()[:80],
                        "seconds": round(time.time() - t0, 1)})

    def do_model_restart(self):
        """Serialise restarts: two overlapping stop+load sequences would race
        on the same port (the second load sees a dying runner and exits early,
        leaving the first one's half-dead process answering)."""
        if not MODEL_RESTART_LOCK.acquire(blocking=False):
            self._json({"ok": False,
                        "detail": "a model restart is already running - wait for it to finish"},
                       409)
            return
        try:
            self._do_model_restart()
        finally:
            MODEL_RESTART_LOCK.release()

    def _do_model_restart(self):
        """Restart the model server. This is the manual workaround for the
        '@@@' degeneration: a fresh runner is always clean."""
        scripts = os.path.join(ROOT, "scripts")
        marker = os.path.join(ROOT, "state", "opencodemodel-unloaded")
        t0 = time.time()
        steps = []
        # STOP, then WAIT for the port to actually go down, then LOAD.
        # Without the wait, llm-load.ps1 sees the old server still answering and
        # exits early ("already running") - so the POISONED runner survives the
        # "restart" and every request after it is still garbage. That race is
        # what made the RESTART button look broken.
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 os.path.join(scripts, "llm-stop.ps1")],
                capture_output=True, text=True, timeout=300)
            steps.append({"step": "llm-stop.ps1", "rc": r.returncode})
            if r.returncode == 3:
                # llm-stop.ps1 refuses to kill a runner that is mid-task. A
                # restart now would abort the operator's running task with
                # ConnectionResetError [WinError 10054] - the exact failure we
                # are fixing. Report it instead of loading on top.
                self._json({"ok": False, "busy": True,
                            "detail": "the model is serving a task right now - restart refused so the task is not "
                                      "aborted. Stop the task first (STOP, then RESTART).",
                            "steps": steps}, 409)
                return
            # Restart is really happening -> the model SHOULD be up, so the
            # unload marker (which blocks the autostart task) can go. Cleared
            # only after a successful stop, so a refused restart leaves it alone.
            try:
                if os.path.isfile(marker):
                    os.remove(marker)
            except Exception:
                pass
        except Exception as e:
            steps.append({"step": "llm-stop.ps1", "error": str(e)[:120]})

        down = False
        for _ in range(40):          # up to 20 s
            try:
                urllib.request.urlopen("http://127.0.0.1:8099/v1/models", timeout=3)
            except Exception:
                down = True
                break
            time.sleep(0.5)
        steps.append({"step": "port closed", "ok": down})
        if not down:
            self._json({"ok": False, "model_up": True,
                        "detail": "the old server would not release the port",
                        "steps": steps})
            return

        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 os.path.join(scripts, "llm-load.ps1")],
                capture_output=True, text=True, timeout=300)
            steps.append({"step": "llm-load.ps1", "rc": r.returncode})
            detail = (r.stdout or "").strip()
            if detail:
                steps.append({"step": "detail",
                              "error": " / ".join(x.strip() for x in detail.splitlines()
                                                  if x.strip())[-500:]})
        except Exception as e:
            steps.append({"step": "llm-load.ps1", "error": str(e)[:120]})
        up = False
        try:
            urllib.request.urlopen("http://127.0.0.1:8099/v1/models", timeout=8)
            up = True
        except Exception:
            up = False
        served = loaded_alias() if up else ""
        want = PROFILES[current_profile()]["alias"]
        if up and served and served != want:
            # the port is open but the WRONG weights are loaded (a stale
            # state\llm-model.cmd, or the old runner never died): say it as a
            # failure instead of a green MODEL UP the agent cannot use.
            steps.append({"step": "model check", "error":
                          "the runner answers as '%s' but profile '%s' asks for '%s'"
                          % (served, current_profile(), want)})
            up = False
        self._json({"ok": up, "model_up": up, "loaded_model": served,
                    "model_expected": want, "seconds": round(time.time() - t0, 1),
                    "steps": steps})


    # --------------------------------------------------------------- POST
    def do_POST(self):
        if self.path.startswith("/api/prompt"):
            body = self._body()
            text = body.get("system")
            if not isinstance(text, str) or not text.strip():
                self._json({"ok": False, "error": "the prompt cannot be empty"}, 400)
                return
            meta = agent_meta()
            try:
                path = str(meta["prompt_file"](CFG))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                _backup(path, CFG)
                config_lib.write_atomic(path, text.rstrip() + "\n")
                self._json({"ok": True, "path": path, "chars": len(text.rstrip()) + 1})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
            return
        if self.path.startswith("/api/memory"):
            body = self._body()
            which = body.get("which")
            text = body.get("text")
            pr = CFG["prompt"]
            if which not in ("memory", "user"):
                self._json({"ok": False, "error": "which must be memory|user"}, 400)
                return
            if not isinstance(text, str):
                self._json({"ok": False, "error": "text must be a string"}, 400)
                return
            path = pr["memory_file"] if which == "memory" else pr["user_file"]
            if not text.strip() and not body.get("force"):
                try:
                    cur = open(path, "r", encoding="utf-8", errors="replace").read()
                except OSError:
                    cur = ""
                if cur.strip():
                    self._json({"ok": False, "kept": len(cur),
                                "error": "refusing to blank a %d-char %s file - "
                                         "send force:true if you really mean it"
                                         % (len(cur), which)}, 409)
                    return
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                _backup(path, CFG)
                config_lib.write_atomic(path, text)
                budget = pr.get("memory_budget", 2200) if which == "memory" else pr.get("user_budget", 1375)
                self._json({"ok": True, "path": path, "chars": len(text),
                            "over_budget": len(text) > int(budget), "budget": budget})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
            return
        if self.path.startswith("/api/config"):
            body = self._body()
            cfg = body.get("config")
            if not isinstance(cfg, dict):
                self._json({"ok": False, "error": "not a config object"}, 400)
                return
            try:
                config_lib.save(config_lib._merge(config_lib.DEFAULTS, cfg))
                globals()["CFG"] = config_lib.load()
                self._json({"ok": True, "path": CONFIG_PATH})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
            return
        if self.path.startswith("/api/session/"):
            action = self.path.rsplit("/", 1)[-1]
            body = self._body()
            sid = (body.get("id") or "").strip()
            if not sid and action != "new":
                self._json({"error": "session id required"}, 400)
                return
            with LOCK:
                busy = bool(STATE.get("busy"))
            if busy:
                self._json({"ok": False, "busy": True,
                            "error": "a task is running - stop it first, or the session file "
                                     "would be overwritten mid-run"}, 409)
                return
            if action == "new":
                if not sid:
                    sid = next_session_name()
                elif clean_id(sid) != sid:
                    self._json({"ok": False, "id": sid, "suggest": clean_id(sid),
                                "error": "a session name can only use letters, digits, dots, "
                                         "dashes and underscores (it is a file name)"}, 400)
                    return
                if os.path.exists(session_file(sid)) and not body.get("overwrite"):
                    self._json({"ok": False, "conflict": True, "id": sid,
                                "error": "session %s already exists - use another name "
                                         "(or pass overwrite:true to wipe it)" % sid}, 409)
                    return
                os.makedirs(STATE_DIR, exist_ok=True)
                d = {"id": sid, "cwd": body.get("cwd", ""), "title": "", "messages": [],
                     "created": time.time(), "updated": time.time(), "compacted": False}
                config_lib.write_atomic(session_file(sid), json.dumps(d, indent=1))
                with LOCK:
                    STATE["session"] = sid
                self._json({"ok": True, "id": sid})
            elif action == "rename":
                to = str(body.get("to") or "").strip()
                src = session_file(sid)
                if not os.path.isfile(src):
                    self._json({"ok": False, "error": "no such session: %s" % sid}, 404)
                    return
                if not to:
                    self._json({"ok": False, "error": "new name required"}, 400)
                    return
                if clean_id(to) != to:
                    self._json({"ok": False, "suggest": clean_id(to),
                                "error": "a session name can only use letters, digits, dots, "
                                         "dashes and underscores (it is a file name)"}, 400)
                    return
                if to == sid:
                    self._json({"ok": True, "id": sid, "unchanged": True})
                    return
                dst = session_file(to)
                if os.path.exists(dst):
                    self._json({"ok": False, "error": "%s already exists" % to}, 409)
                    return
                try:
                    d = json.loads(_read_text_shared(src))
                    d["id"] = to
                    d["updated"] = time.time()
                    config_lib.write_atomic(dst, json.dumps(d, indent=1, ensure_ascii=False))
                    _retry_os(os.remove, src)
                    lsrc, ldst = session_lock_path(sid), session_lock_path(to)
                    if os.path.isfile(lsrc):
                        _retry_os(os.replace, lsrc, ldst)
                    with LOCK:
                        if STATE.get("session") == sid:
                            STATE["session"] = to
                    self._json({"ok": True, "id": to, "old": sid})
                except Exception as e:
                    self._json({"ok": False, "error": str(e)}, 500)
            elif action == "reset":
                d = {"id": sid, "cwd": body.get("cwd", ""), "title": "", "messages": [],
                     "created": time.time(), "updated": time.time(), "compacted": False}
                os.makedirs(STATE_DIR, exist_ok=True)
                with open(session_file(sid), "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=1)
                self._json({"ok": True, "id": sid})
            elif action == "compact":
                try:
                    rc, out, err = run_agent(["--session", sid, "--compact", "--json"], timeout=600)
                    events = []
                    for line in (out or "").splitlines():
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            pass
                    c = next((e for e in events if e.get("type") == "compact"), None)
                    if c and c.get("changed"):
                        self._json({"ok": True, "summary": c.get("summary", "")[:2000],
                                    "before": c.get("before"), "after": c.get("after"),
                                    "tokens": c.get("tokens")})
                    else:
                        self._json({"ok": True, "reason": (c or {}).get("reason", "nothing to compact")})
                except Exception as e:
                    self._json({"error": str(e)}, 500)
            else:
                self._json({"error": "unknown action"}, 404)
            return

        if self.path.startswith("/api/cancel"):
            with LOCK:
                proc = STATE.get("proc")
                if proc is not None and proc.poll() is None:
                    STATE["cancelled"] = True
            if proc is None or proc.poll() is not None:
                self._json({"ok": True, "detail": "nothing was running"})
                return
            # kill the process TREE: the agent spawns the model calls, and on
            # Windows a plain terminate() can leave the child behind
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=60)
            self._json({"ok": True, "detail": f"stopped the agent (pid {proc.pid})"})
            return

        if self.path.startswith("/api/profile"):
            body = self._body()
            key = str(body.get("id") or "").strip()
            if key not in PROFILES:
                self._json({"error": "unknown profile '%s'" % key,
                            "profiles": list(PROFILES)}, 400)
                return
            prof = set_profile(key)
            try:
                cfg = config_lib.load()
                cfg.setdefault("agent", {})
                cfg["agent"]["ctx_size"] = int(PROFILES[key]["ctx"])
                cfg["agent"]["model"] = PROFILES[key]["alias"]
                config_lib.save(config_lib._merge(config_lib.DEFAULTS, cfg))
                CFG.clear()
                CFG.update(config_lib.load())
            except Exception:
                pass
            self._json({"ok": True, "profile": prof,
                        "detail": "saved - restart the model to load it"})
            return

        if self.path.startswith("/api/tuning"):
            body = self._body()
            try:
                val = int(body.get("budget"))
            except Exception:
                self._json({"error": "budget must be an integer"}, 400)
                return
            stored = set_tuning_budget(val)
            self._json({"ok": True, "budget": stored,
                        "detail": "saved - the model must restart to apply it"})
            return

        if self.path.startswith("/api/model/"):
            action = self.path.rsplit("/", 1)[-1]
            if action == "restart":
                return self.do_model_restart()
            if action == "check":
                return self.do_model_check()
            self._json({"error": "unknown action"}, 404)
            return

        if not self.path.startswith("/api/run"):
            self._json({"error": "not found"}, 404)
            return

        body = self._body()
        task = str(body.get("task") or "").strip()
        if not task:
            # otherwise the agent starts, argparse rejects the empty positional
            # and the log fills with a usage block
            self._json({"error": "the task is empty - type something to run"}, 400)
            return
        cwd = str(body.get("cwd") or "").strip() or os.getcwd()
        session = str(body.get("session") or "").strip()
        try:
            steps = int(body.get("steps") or 8)
        except Exception:
            steps = 8
        steps = max(1, min(40, steps))     # same range as the UI input: 1..40

        with LOCK:
            if STATE["busy"]:
                self._json({"error": "a job is already running"}, 409)
                return
            STATE["busy"] = True
            STATE["task"] = task
            STATE["cwd"] = cwd
            STATE["session"] = session
            STATE["cancelled"] = False
        STATE["started"] = time.time()

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        client_gone = [False]

        def emit(obj):
            # If the browser closed the tab, stop writing to the socket but
            # KEEP RUNNING the agent: it still has to finish and save the
            # session. Killing it here is what used to lose the conversation.
            if client_gone[0]:
                return
            chunk = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
            try:
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            except Exception:
                client_gone[0] = True

        proc = None
        try:
            prof = PROFILES[current_profile()]
            # This path builds its own command line (it streams NDJSON), so it
            # must carry the profile too: llama-server rejects a model id it does
            # not serve, and --ctx-size drives the agent's context estimate.
            cmd = [find_python(), AGENT, "--config", CONFIG_PATH, "--cwd", cwd,
                   "--max-steps", str(steps), "--model", prof["alias"],
                   "--ctx-size", str(prof["ctx"]), "--json"]
            if session:
                cmd += ["--session", session]
            # "--" ends option parsing: a task that starts with - is still a task
            cmd += ["--", task]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            with LOCK:
                STATE["proc"] = proc
            agent_said_done = [False]
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    # a raw line the agent could not JSON-parse: it is usually
                    # a traceback. Show it instead of swallowing it.
                    emit({"type": "error", "message": line.strip()[:400]})
                    continue
                if ev.get("type") == "done":
                    agent_said_done[0] = True
                emit(ev)
            proc.wait()
            # the agent already reports its own outcome; only add the exit code
            # when it died before it could say anything
            if not agent_said_done[0]:
                with LOCK:
                    was_cancelled = STATE.get("cancelled", False)
                if was_cancelled:
                    emit({"type": "done", "reason": "cancelled by the user",
                          "failed": False})
                elif proc.returncode == 0:
                    emit({"type": "done", "reason": "finished", "failed": False})
                else:
                    emit({"type": "done",
                          "reason": f"the agent crashed (exit {proc.returncode})",
                          "failed": True})
        except Exception as e:
            emit({"type": "error", "message": str(e)})
        finally:
            # The agent is left to finish on its own if the client vanished.
            # It saves the session itself after every step.
            if proc and proc.poll() is None and not client_gone[0]:
                try:
                    proc.kill()
                except Exception:
                    pass
            with LOCK:
                STATE["busy"] = False
                STATE["proc"] = None
            try:
                self.wfile.write(b"0\r\n\r\n")
            except Exception:
                pass


if __name__ == "__main__":
    if not os.path.isfile(AGENT):
        sys.exit(f"missing agent script: {AGENT}")
    print(f"mini-agent web -> http://127.0.0.1:{PORT}")
    class QuietServer(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # A browser tab that reloads or closes mid-response raises
            # ConnectionReset/ConnectionAborted out of socketserver. Printing a
            # 25-line traceback for each one buried the real errors (and once
            # made a killed model look like a web-UI crash), so say nothing.
            etype = sys.exc_info()[0]
            if etype and issubclass(etype, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)

    QuietServer(("127.0.0.1", PORT), Handler).serve_forever()
