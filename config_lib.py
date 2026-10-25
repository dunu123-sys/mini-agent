"""config_lib — one JSON file, defaults in code, used by BOTH v2 scripts.

Why a separate file: the agent and the page must never disagree about a budget
(the old pair drifted: the page hardcoded 32768 while the runner was at 16384).
Everything tunable lives in config.json next to this file; missing keys fall back
to DEFAULTS so a half-written file still runs.
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

DEFAULTS = {
    "agent": {
        "script": "mini-agent.py",
        "server": "http://127.0.0.1:8099/v1",
        "model": "qwen35-9b",
        "ctx_size": 16384,
        "max_steps": 12,
        "max_tokens": 6144,
        "temperature": 0.3,
        "overhead_tokens": 900,
        "retry_cut_tool_call": True,
    },
    "tools": {
        "max_result_chars": 5000,
        "bash_tail_ratio": 0.7,
        "step_cap_multiplier": 3,
        "read": {"line_numbers": True, "max_lines": 400, "outline_over_lines": 200,
                 "head_lines": 120, "tail_lines": 60, "outline_max_items": 80},
        "grep": {"max_matches": 40, "context": 2, "max_line_chars": 200,
                 "skip_dirs": [".git", "node_modules", "__pycache__", ".venv", "venv",
                               "dist", "build", ".next"],
                 "max_file_bytes": 2000000},
        "bash": {"timeout_s": 30, "shell": "", "kill_tree": True},
        "cache": {"reads": True},
    },
    "compaction": {
        "auto": True, "trigger_pct": 0.85, "keep_last_messages": 6,
        "keep_last_tokens": 2000, "trim_tool_results_first": True,
        "trim_tool_result_chars": 1200, "digest_chars": 12000, "summary_max_tokens": 900,
    },
    "prompt": {
        "system_file": "prompts/system.md",
        "inject_memory": False,
        "memory_file": "memories/MEMORY.md",
        "user_file": "memories/USER.md",
        "memory_max_chars": 4000,
        "memory_budget": 2200,
        "user_budget": 1375,
        "backup_on_save": True,
    },
    "ui": {"port": 8795, "sessions_dir": "state/sessions",
           "title": "MINI AGENT v2", "poll_ms": 3000, "auto_scroll_px": 90},
}



def atomic_replace(src, dst):
    """Put src in place of dst even if a reader has dst open.

    os.replace() is MoveFileEx(REPLACE_EXISTING) and answers WinError 5 while
    any handle is open on the destination - even one that requested
    FILE_SHARE_DELETE (measured on Windows 10 19044). ReplaceFileW honours the
    share mode, so use it whenever dst already exists.
    """
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.ReplaceFileW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR,
                                         wintypes.LPCWSTR, wintypes.DWORD,
                                         ctypes.c_void_p, ctypes.c_void_p]
            k32.ReplaceFileW.restype = wintypes.BOOL
            if k32.ReplaceFileW(dst, src, None, 0, None, None):
                return
            err = ctypes.get_last_error()
            if err != 2:                     # 2 = dst does not exist yet
                raise OSError(err, "ReplaceFileW failed", dst)
        except OSError:
            raise
        except Exception:
            pass
    os.replace(src, dst)


def write_atomic(path, text, retries=8):
    """Write a file the other process may be reading at the same time.

    Used for everything the agent loads while it starts (config.json,
    prompts/system.md, MEMORY.md, USER.md): a plain truncate-write there means
    the agent can parse half a file, and config_lib.load() would silently fall
    back to DEFAULTS (wrong model, wrong context) - or crash.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    last = None
    for i in range(retries):
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            atomic_replace(tmp, path)
            return True
        except OSError as e:
            last = e
            time.sleep(0.03 * (i + 1))
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return True
    except OSError:
        pass
    raise last or OSError("could not write %s" % path)


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path=None):
    p = path or CONFIG_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    return _merge(DEFAULTS, data)


def save(cfg, path=None):
    p = path or CONFIG_PATH
    write_atomic(p, json.dumps(cfg, indent=2))


def get(cfg, dotted, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
