#!/usr/bin/env python
"""
mini-agent — a minimal coding-agent harness for the local qwen3.5 model.

WHY THIS EXISTS
    OpenCode's prompt is ~31 KB (9.8 KB system + 21 KB of tool schemas) and the
    qwen3.5 4B on an RX 580X prefills that at ~50-150 tok/s, so every turn
    spends 1-3 minutes just reading the prompt before the model thinks. It also
    keeps a 10-tool surface, which is more than this model can use reliably.

    This harness is the "Pi" idea in ~250 lines: 4 tools, a ~400-byte system
    prompt, one loop. It talks to the same llama-server on :8099.

    Measured prompt sizes:
        OpenCode : ~31 KB  (~7,800 tokens)  -> 50-150 s of prefill per turn
        this     : ~2.4 KB (~600 tokens)    -> 3-8 s of prefill per turn

SESSIONS
    A session is a named, persistent conversation stored as JSON in the sessions
    directory from config.json (ui.sessions_dir), one file per session.
    Pass --session to continue one, --reset to wipe it, --compact to shrink it.

        python mini-agent.py --session work "start the task"
        python mini-agent.py --session work "next thing"      # continues
        python mini-agent.py --session work --reset            # wipes history
        python mini-agent.py --session work --compact          # shrinks history
        python mini-agent.py --list-sessions

USAGE
    python mini-agent.py "list the files here and tell me the python ones"
    python mini-agent.py --cwd F:/Codes/something "add a docstring to main.py"
    python mini-agent.py --json          # machine-readable events on stdout
"""
import argparse
import fnmatch
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config_lib

CFG = config_lib.load()
STATE_DIR = CFG["ui"]["sessions_dir"]
CONFIG_PATH = config_lib.CONFIG_PATH

# Read cache for THIS process: path -> (mtime, size, lines). A second read of an
# unchanged file costs one line instead of 5 000 chars (measured: the same HTML
# was read twice per session).
READS = {}
RUN_LOCKS = set()

# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file with line numbers. For a big file prefer mode=outline (structure with line numbers, cheap) or a start/end range. Reading the same unchanged file twice returns a short notice instead of the text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "description": "first line, 1-based"},
                    "end": {"type": "integer", "description": "last line, inclusive"},
                    "mode": {"type": "string", "enum": ["text", "outline"],
                             "description": "outline = sections, ids, functions and classes with line numbers"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file, creating directories as needed. For long files (more than about 120 lines) write a short skeleton first and append the rest with edit calls - one huge content string gets cut off and the call fails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append",
            "description": "Append text to the end of a file (creates it). Use this to build a long file in several calls: write the first section, then append the rest one section at a time.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace an exact string in a file. old_string must appear exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents. Returns path:line: text (cheap - use this before reading a file).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regular expression"},
                    "path": {"type": "string", "description": "file or directory (default .)"},
                    "glob": {"type": "string", "description": "e.g. *.html or *.py"},
                    "context": {"type": "integer", "description": "lines before/after each hit"},
                    "max": {"type": "integer", "description": "stop after N matches"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list",
            "description": "List files under a directory (name, size, mtime). Use instead of shelling out to dir/ls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "glob": {"type": "string", "description": "e.g. *.py (default *)"},
                    "recursive": {"type": "boolean"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return its output (tail-biased: errors are at the end). The shell is Windows cmd.exe: `&` just separates commands, it does NOT run anything in the background - a server started that way blocks until the timeout. Use `start /b <cmd>` (or spawn it and move on) to leave a server running. Prefer grep/list/read for file work.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"},
                               "timeout_s": {"type": "integer",
                                             "description": "optional; default from config.json (tools.bash.timeout_s)"}},
                "required": ["command"],
            },
        },
    },
]

SYSTEM = """You are a coding assistant working in a terminal on Windows.

Tools: read, grep, list, write, edit, append, bash. Use them directly; do not
describe what you would do. Read before you edit. Keep answers short. When the
task is done, state the result in plain text and stop calling tools.

Work cheaply: grep or list to FIND, read a range (start/end) to SEE, read with
mode="outline" to get a file's structure (functions, ids, line numbers) instead
of the whole file. Never read a big file twice - if it says unchanged, ask for a
range instead.

The shell is Windows cmd.exe: `&` does not background anything, use
`start /b <cmd>` to leave a server running.

Tool arguments are capped by the reply length. Build long files in SECTIONS:
write the first ~60 lines, then append the next ~60, repeating until done. Never
put a whole large file in one call - the reply is cut mid-string and the call is
rejected by the server."""

# --------------------------------------------------------------------------
# prompt: an editable file + optional memory injection
# --------------------------------------------------------------------------
def prompt_file(cfg=None):
    cfg = cfg or CFG
    p = cfg["prompt"]["system_file"]
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def _read_text(path, limit=None):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            t = f.read()
        return t if limit is None else t[:limit]
    except OSError:
        return ""


def load_system(cfg=None):
    """The system message. Read from prompts/system.md so the page can edit it,
    falling back to the built-in text; memory blocks are OPT-IN (inject_memory)
    because they are the operator's real MEMORY.md/USER.md, not the agent's."""
    cfg = cfg or CFG
    system = _read_text(prompt_file(cfg)).strip() or SYSTEM
    pr = cfg.get("prompt") or {}
    if pr.get("inject_memory"):
        cap = int(pr.get("memory_max_chars") or 4000)
        mem = _read_text(pr.get("memory_file") or "").strip()
        usr = _read_text(pr.get("user_file") or "").strip()
        if mem or usr:
            system += ("\n\n--- MEMORY (durable notes about this host and the operator) ---\n"
                       + mem[:cap])
            if usr:
                system += "\n\n--- OPERATOR (who you are working for) ---\n" + usr[:cap]
            system += "\n--- END MEMORY ---"
    return system


# rough tokens-per-char for English/code on a BPE tokenizer
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text):
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def estimate_context(messages):
    """Approximate the prompt size the server will have to prefill."""
    if not messages:
        return 0
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += estimate_tokens(fn.get("name", "")) + estimate_tokens(fn.get("arguments", ""))
        total += 8  # role + framing overhead per message
    # The old version returned total+60 and ignored the system prompt AND the tool
    # schemas (~850 tokens), so "room" looked bigger than it was and a step could
    # overflow the real window.
    return total + int(CFG["agent"]["overhead_tokens"])


# --------------------------------------------------------------------------
# session storage
# --------------------------------------------------------------------------
def session_path(sid):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid)[:64] or "default"
    return os.path.join(STATE_DIR, safe + ".json")


def blank_session(sid):
    return {"id": sid, "cwd": "", "title": "", "messages": [],
            "created": time.time(), "updated": time.time(), "compacted": False}


def load_session(sid):
    p = session_path(sid)
    if not os.path.isfile(p):
        return blank_session(sid)
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        d.setdefault("messages", [])
        d.setdefault("id", sid)
        return d
    except Exception:
        return blank_session(sid)


def repair_dangling_tools(messages):
    """A run stopped while a tool was executing leaves an assistant message whose
    tool_calls have no answers. llama-server rejects that history on the next
    request ("tool_calls must be followed by tool messages"), so fill the blanks
    with a note instead of resuming a broken conversation."""
    if not messages:
        return messages
    last, idx = None, None
    for k in range(len(messages) - 1, -1, -1):
        if messages[k].get("role") == "assistant":
            last, idx = messages[k], k
            break
    if not last or not last.get("tool_calls"):
        return messages
    answered = {m.get("tool_call_id") for m in messages[idx + 1:]
                if isinstance(m, dict) and m.get("role") == "tool"}
    for tc in last["tool_calls"]:
        if (tc.get("id") or "") not in answered:
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or "",
                             "content": "(interrupted: this tool call never returned - "
                                        "the run was stopped while it was running. "
                                        "Repeat it if it is still needed.)"})
    return messages



def _atomic_replace(src, dst):
    """Atomically put src in place of dst, even if someone is reading dst.

    os.replace() is MoveFileEx(REPLACE_EXISTING) and it answers WinError 5 while
    *any* handle is open on the destination - even one that asked for
    FILE_SHARE_DELETE (measured on this box, Win10 19044). ReplaceFileW() is the
    API that honours the share mode, so use it and keep os.replace only for the
    case where the destination does not exist yet.
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
            if err != 2:                 # 2 = dst does not exist yet
                raise OSError(err, "ReplaceFileW failed", dst)
        except OSError:
            raise
        except Exception:
            pass                          # ctypes unavailable: fall through
    os.replace(src, dst)


def save_session(sess):
    """Atomic save that survives a concurrent reader on Windows.

    os.replace() is MoveFileEx(REPLACE_EXISTING): it needs DELETE access on the
    destination, and Python's open() grabs the file with FILE_SHARE_READ|WRITE
    only. The web page polls /api/sessions every few seconds, so a save that
    lands exactly while a reader has the file open dies with
    PermissionError [WinError 5] - which used to crash the whole run.
    Retry briefly, then fall back to a plain in-place write (readers open with
    share-deny-none, so truncate-write is allowed) instead of raising.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    sess["updated"] = time.time()
    target = session_path(sess["id"])
    tmp = target + ".tmp"
    data = json.dumps(sess, indent=1, ensure_ascii=False)
    last = None
    for attempt in range(8):       # ~1.1 s total: a reader holds the file
        try:                        # for microseconds, not seconds
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
            _atomic_replace(tmp, target)
            return True
        except OSError as e:          # PermissionError is an OSError on Windows
            last = e
            time.sleep(0.03 * (attempt + 1))
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(data)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return True
    except OSError as e:
        last = e
    print("warning: could not save session %s (%s) - the run continues"
          % (sess["id"], last), file=sys.stderr, flush=True)
    return False


def list_sessions():
    out = []
    if not os.path.isdir(STATE_DIR):
        return out
    for fn in os.listdir(STATE_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(STATE_DIR, fn), "r", encoding="utf-8") as f:
                d = json.load(f)
            out.append({
                "id": d.get("id", fn[:-5]),
                "title": d.get("title", ""),
                "cwd": d.get("cwd", ""),
                "messages": len(d.get("messages") or []),
                "tokens": estimate_context(d.get("messages") or []),
                "updated": d.get("updated", 0),
            })
        except Exception:
            continue
    out.sort(key=lambda x: x.get("updated", 0), reverse=True)
    return out


# --------------------------------------------------------------------------
# tool implementations
# --------------------------------------------------------------------------
def _resolve(path, cwd):
    p = os.path.join(cwd, path) if not os.path.isabs(path) else path
    return os.path.normpath(p)


# ---- tool-output budget -----------------------------------------------------
# ONE tool call can return more text than the entire conversation: `bash` was
# allowed 20 000 chars (~5 k tokens) and `read` had a line cap but no byte cap,
# so 200 lines of minified JS is ~50 k tokens. Measured consequence: a single
# result pushed a run to a 10 240-token prompt prefill (59-85 t/s on this card)
# and buried the task in noise. Every result now passes through _clip_tool(),
# so a tool added later cannot bypass the budget.
MAX_TOOL_CHARS = 5000


def _clip_tool(text, limit=None, tail_ratio=None):
    """Head+tail clip that never lies and never silently truncates the tail.

    Three defects this replaces (all reproduced on real sessions):
      - the old version advertised "first 120 and last 60 lines" and then cut the
        result to 5 000 chars, so the model saw ~114 lines of a 431-line file;
      - when the marker pushed the body over the limit it did body[:limit], which
        chopped the TAIL mid-line with no hint;
      - one head/tail ratio for every tool, although build logs need the tail.
    So: reserve the marker first, cut on line boundaries, and say exactly what
    was dropped. Returns (text, original_chars).
    """
    limit = CFG["tools"]["max_result_chars"] if limit is None else int(limit)
    n = len(text)
    if n <= limit:
        return text, n

    ratio = CFG["tools"]["bash_tail_ratio"] if tail_ratio is None else float(tail_ratio)
    ratio = min(0.9, max(0.1, ratio))
    marker = ("\n...[TRUNCATED: %d of %d chars dropped. Kept the head and the tail; "
              "use grep / list / read start=end= / read mode=outline for the rest]...\n")
    # worst case marker length (the numbers are at most 12 digits each)
    marker_room = 190
    body_budget = max(200, limit - marker_room)
    head_chars = int(body_budget * (1.0 - ratio))
    tail_chars = body_budget - head_chars

    head = text[:head_chars]
    cut = head.rfind("\n")
    if cut > head_chars * 0.5:          # keep whole lines, but do not lose half the head
        head = head[:cut + 1]
    tail = text[-tail_chars:] if tail_chars else ""
    cut = tail.find("\n")
    if 0 <= cut < tail_chars * 0.5:
        tail = tail[cut + 1:]

    body = head + (marker % (n - len(head) - len(tail), n)) + tail
    if len(body) > limit:               # never silently drop the end: shrink the head
        over = len(body) - limit
        body = body[over:] if over < len(head) else body[:limit]
    return body, n


# --------------------------------------------------------------------------
# read: line numbers, ranges, outline, and a per-run cache
# --------------------------------------------------------------------------
OUTLINE_RULES = [
    # (regex, label) - cheap and language-agnostic: this is a MAP, not a parser
    (re.compile(r"^.{0,12}<script\b", re.I), "script start"),
    (re.compile(r"^.{0,12}</script>", re.I), "script end"),
    (re.compile(r"^.{0,12}<style\b", re.I), "style start"),
    (re.compile(r"^.{0,12}</style>", re.I), "style end"),
    (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"), "function"),
    (re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?[\(A-Za-z_$]"), "binding"),
    (re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\("), "def"),
    (re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[\(:]"), "class"),
    (re.compile(r"^\s*@[A-Za-z_][\w.]*"), "decorator"),
    (re.compile(r"<section[^>]*\bid=[\"']([^\"']+)[\"']"), "section"),
    (re.compile(r"<div[^>]*\bid=[\"']([^\"']+)[\"']"), "div id"),
    (re.compile(r"<title>([^<]*)</title>", re.I), "title"),
]


def _outline(path, lines, max_items):
    """A structure index: line number + what lives there. ~5x cheaper than the
    text it replaces, and it keeps the entries that matter: script blocks and
    functions are never collapsed, runs of ids are."""
    ext = os.path.splitext(path)[1].lower()
    raw_items = []
    for no, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        for rx, label in OUTLINE_RULES:
            m = rx.search(line)
            if not m:
                continue
            name = (m.group(1) if m.lastindex else "").strip()
            if label in ("binding", "decorator") and ext not in (".js", ".mjs", ".ts", ".py"):
                break
            raw_items.append((no, label, name[:70] if name else line.strip()[:70]))
            break

    # collapse consecutive same-label runs (an HTML with 60 tab ids turned the
    # whole map into "div id" noise)
    items, k = [], 0
    while k < len(raw_items):
        no, label, name = raw_items[k]
        if label in ("div id", "section"):
            run = [raw_items[k]]
            while (k + len(run) < len(raw_items)
                   and raw_items[k + len(run)][1] == label
                   and raw_items[k + len(run)][0] == run[-1][0] + 1):
                run.append(raw_items[k + len(run)])
            if len(run) >= 4:
                items.append("  %5d| %-11s x%d  (%s .. %s)" % (no, label, len(run),
                                                               run[0][2], run[-1][2]))
                k += len(run)
                continue
        items.append("  %5d| %-11s %s" % (no, label, name))
        k += 1
        if len(items) >= max_items:
            items.append("  ... (map stops at %d items)" % max_items)
            break
    return "MAP" + ("" if items else " (no structure found)") + "\n" + "\n".join(items)


def _numbered(lines, start, end):
    return "\n".join("%5d| %s" % (no, lines[no - 1].rstrip("\n")) for no in range(start, end + 1))


def tool_read(args, cwd):
    raw = (args.get("path") or "").strip()
    if not raw:
        return "error: path is required"
    p = _resolve(raw, cwd)
    if os.path.isdir(p):
        return ("error: %s is a DIRECTORY. Use list(path=\"%s\")." % (raw, raw))
    if not os.path.isfile(p):
        base = os.path.basename(raw or ".")
        near = []
        try:
            d = os.path.dirname(p) or "."
            if os.path.isdir(d):
                near = [f for f in os.listdir(d) if base.lower()[:4] in f.lower()][:5]
        except OSError:
            pass
        return ("error: not a file: %s%s" % (raw, ("; nearby: " + ", ".join(near)) if near else ""))
    cfg = CFG["tools"]["read"]
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        return "error: %s" % e
    lines = text.splitlines()
    total = len(lines)
    if not total:
        return "%s is empty (0 lines)" % raw

    mode = (args.get("mode") or "text").lower()
    start = args.get("start")
    end = args.get("end")
    numbered = cfg["line_numbers"]

    # --- outline requested -------------------------------------------------
    if mode == "outline":
        head = "[%d lines] %s\n" % (total, raw)
        return head + _outline(p, lines, cfg["outline_max_items"]) + \
               "\n\n(next: read start=<line> end=<line> to see a section)"

    # --- explicit range
    max_lines = int(cfg["max_lines"])
    if start is not None or end is not None:
        try:
            s = max(1, int(start or 1))
            e = min(total, int(end or min(total, s + max_lines - 1)))
        except (TypeError, ValueError):
            return "error: start/end must be integers"
        if e < s:
            return "error: end (%d) is before start (%d)" % (e, s)
        note = ""
        if e - s + 1 > max_lines:
            e = s + max_lines - 1
            note = "\n[window capped at %d lines - call again with start=%d]" % (max_lines, e + 1)
        body = _numbered(lines, s, e) if numbered else "\n".join(lines[s - 1:e])
        return "[%d lines total, showing %d-%d]%s\n%s" % (total, s, e, note, body)

    # --- whole file --------------------------------------------------------
    # Cache covers WHOLE reads only: an outline or a range must never make the
    # next full read answer "unchanged" (it never showed the text).
    key = os.path.abspath(p)
    try:
        sig = (os.stat(p).st_mtime_ns, os.stat(p).st_size)
    except OSError:
        sig = None
    prev = READS.get(key)
    READS[key] = sig
    if CFG["tools"]["cache"]["reads"] and sig is not None and prev == sig:
        return ("%s is UNCHANGED since you read it in this run (%d lines). Do not read it "
                "again - ask for a range (start/end) or mode=outline if you need a "
                "specific part." % (raw, total))

    if total <= int(cfg["outline_over_lines"]):
        body = _numbered(lines, 1, total) if numbered else "\n".join(lines)
        return "[%d lines]\n%s" % (total, body)

    # Big file: MAP first (it always survives the clipper), then the head.
    head_n = min(int(cfg["head_lines"]), total)
    body = ("[%d lines: MAP first, then lines 1-%d. Use start=/end= for any other range, "
            "mode=outline for the map alone]\n" % (total, head_n)
            + _outline(p, lines, 40) + "\n\nCODE 1-%d\n" % head_n
            + (_numbered(lines, 1, head_n) if numbered else "\n".join(lines[:head_n])))
    return body


def _write_text(path, text):
    """Write a file the way Windows actually fails. Returns (ok, message).

    Failure modes that ALL read as "[Errno 13] Permission denied":
      - the read-only ATTRIBUTE (fixed by chmod), and
      - a LOCK: another process holds the file open (Next dev server, editor,
        indexer). WinError 32 maps to errno 13, so a transient lock looks like a
        permissions problem.
    Order: chmod -> retry with backoff -> atomic replace -> sibling file.
    A locked file cannot be replaced at all on Windows (MoveFileEx needs DELETE
    sharing, which watchers do not grant), so the last resort writes beside it
    instead of throwing the content away.
    """
    last = None
    if os.path.exists(path):
        try:
            mode = os.stat(path).st_mode
            if not (mode & stat.S_IWRITE):
                os.chmod(path, mode | stat.S_IWRITE)
        except OSError:
            pass
    for attempt in range(4):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return True, ""
        except PermissionError as e:
            last = e
            time.sleep(0.4 * (attempt + 1))   # watcher locks are usually brief
        except OSError as e:
            last = e
            break
    tmp = path + ".agent-tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        _atomic_replace(tmp, path)
        return True, ""
    except OSError as e:
        last = last or e
        try:
            os.remove(tmp)
        except OSError:
            pass
    if not os.path.exists(path):
        return False, ("error: could not create %s (%s). Check the folder exists and is "
                       "writable." % (path, last))
    root, ext = os.path.splitext(path)
    alt = "%s.agent%d%s" % (root, int(time.time() * 1000) % 100000000, ext)
    try:
        with open(alt, "w", encoding="utf-8") as f:
            f.write(text)
        return True, ("note: %s is locked by another process, so the content went to %s "
                      "instead - rename it over the original when the lock clears."
                      % (os.path.basename(path), os.path.basename(alt)))
    except OSError:
        pass
    return False, ("error: could not write %s (%s). The file is held open by another "
                   "process - a dev server, an editor or an indexer - and even a sibling "
                   "file could not be created. Use a different folder." % (path, last))


def tool_write(args, cwd):
    raw = (args.get("path") or "").strip()
    if not raw:
        return "error: path is required (give a FILE path, e.g. projects/index.html)"
    p = _resolve(raw, cwd)
    content = args.get("content", "")
    # A directory target fails as "[Errno 13] Permission denied" on Windows,
    # which looks like a lock or an ACL. It is neither: the model named a folder.
    if os.path.isdir(p):
        return ("error: %s is a DIRECTORY, not a file. Give a file path inside it, "
                "e.g. %s" % (raw, os.path.join(raw.rstrip("/\\"), "index.html").replace("\\", "/")))
    if not content:
        return ("error: no content supplied, nothing was written. Send the file text in "
                "the content field (or build the file in parts with write then append).")
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    except OSError as e:
        return f"error: {e}"
    ok, msg = _write_text(p, args.get("content", ""))
    if not ok:
        return msg
    if msg:
        return msg + " (%d chars)" % len(args.get("content", ""))
    return f"wrote {len(args.get('content',''))} chars to {args.get('path')}"


def tool_edit(args, cwd):
    p = _resolve(args.get("path", ""), cwd)
    old = args.get("old_string", "")
    new = args.get("new_string", "")
    if not os.path.isfile(p):
        return f"error: not a file: {args.get('path')}"
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        n = txt.count(old)
        if n == 0:
            return "error: old_string not found"
        if n > 1:
            return f"error: old_string appears {n} times; must be unique"
        ok, msg = _write_text(p, txt.replace(old, new, 1))
        if not ok:
            return msg
        return msg or f"edited {args.get('path')}"
    except Exception as e:
        return f"error: {e}"


def _bash_output(path, limit):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
    except Exception:
        return ""
    if len(data) <= limit:
        return data
    # tail-biased: a command that fails says why at the end
    keep = CFG["tools"].get("bash_tail_ratio", 0.7)
    head = int(limit * (1 - keep))
    return data[:head] + "\n[... clipped ...]\n" + data[-int(limit * keep):]


def _port_busy_hint(out):
    """A leftover server squatting the port is why a start fails with WinError
    10048; naming the holder stops the model from hunting for PIDs."""
    if "10048" not in out and "Only one usage of each socket address" not in out:
        return out
    return out.rstrip() + ("\n\n(hint: that port is already held by another process. "
                           "`netstat -ano | findstr :PORT` prints the PID that owns it; "
                           "pick another port or stop that PID.)")


def tool_bash(args, cwd):
    """Shell command that can never freeze the run on a server.

    Three mechanisms, in the order they bit us:
      * output goes to a TEMP FILE, not a pipe: a background child holds a pipe
        open and the read blocks forever;
      * IDLE WATCHDOG: no new output for tools.bash.idle_s seconds while the
        process is alive -> return at once and LEAVE IT RUNNING. That is what
        `python start_server.py`, `python -m http.server 8000` and `npm run dev`
        need, and it is what `start /b` was being asked to do;
      * hard timeout (tools.bash.timeout_s) for commands that keep printing;
        that one is killed with its tree, because subprocess.run(timeout=)
        kills only cmd.exe and leaves the server holding the port.
    """
    cmd = args.get("command", "")
    if not cmd.strip():
        return "error: empty command"
    # cmd.exe CANNOT pass an argument that contains newlines: `python -c "line1
    # line2"` exits 0 in 0.2 s and runs NOTHING (measured 2026-10-25 with a
    # multi-line CORS server the model wrote - the tool said "still running in
    # the background", the port was never opened). Refuse it instead of
    # reporting a success that never happened.
    if re.search(r'-c\s+"[^"]*\n', cmd) or re.search(r"-c\s+'[^']*\n", cmd):
        return ("error: refused - cmd.exe cannot pass a multi-line string to -c. "
                "Nothing was started. Write the code to a .py file and run "
                "`start /b python that.py`.")
    cfg = CFG["tools"]["bash"]
    timeout = max(1, min(1800, int(args.get("timeout_s") or cfg["timeout_s"])))
    idle = max(1.0, float(args.get("idle_s") or cfg.get("idle_s", 15)))
    fd, path = tempfile.mkstemp(prefix="mini-agent-bash-", suffix=".log")
    os.close(fd)
    kw = dict(cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
              stdin=subprocess.DEVNULL)
    if cfg.get("shell"):            # e.g. "C:/Program Files/Git/bin/bash.exe"
        kw["executable"] = cfg["shell"]
    started = time.time()
    timed_out = False
    detached = False
    last_size = 0
    last_change = started
    p = None
    try:
        with open(path, "wb") as fh:
            kw["stdout"] = fh
            kw["stderr"] = subprocess.STDOUT
            try:
                p = subprocess.Popen(cmd, shell=True, **kw)
            except Exception as e:
                return "error: %s" % e
            while True:
                if p.poll() is not None:
                    break
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = last_size
                now = time.time()
                if size > last_size:
                    last_size = size
                    last_change = now
                if now - last_change >= idle:
                    detached = True          # it is a server: leave it up
                    break
                if now - started >= timeout:
                    timed_out = True
                    break
                time.sleep(0.2)
            if timed_out:
                if cfg.get("kill_tree") and os.name == "nt":
                    try:
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                                       capture_output=True, timeout=15)
                    except Exception:
                        p.kill()
                else:
                    p.kill()
                try:
                    p.wait(timeout=10)
                except Exception:
                    pass
    finally:
        pass
    out = _port_busy_hint(_bash_output(path, 200000).strip())
    if not detached:
        try:
            os.remove(path)
        except Exception:
            pass
    secs = time.time() - started
    if timed_out:
        return ("error: command timed out after %ds while still printing output "
                "(process tree killed). If it was meant to be a server, launch it "
                "with `start /b <cmd>` and it will be left running. "
                "Partial output:\n%s" % (timeout, out[-1500:]))
    if detached:
        return ("%s\n\n(still running: no new output for %.0fs, so the command was "
                "left running in the background (shell pid %s) - this is NOT an "
                "error. Check it with netstat/tasklist; start servers with "
                "`start /b`.)" % ((out or "(no output)")[-3000:], secs, p.pid))
    if out:
        return out
    if secs < 2 and re.search(r"(^|[&|(]\s*)start\b", cmd, re.I):
        # a `start`ed command returns as soon as cmd.exe forks it: exit 0 here
        # proves nothing about the program. Say so, and say how to check.
        return ("(no output, exit %s, %.1fs) - `start` returns at once, so this "
                "only means cmd.exe finished: it does NOT prove the program is "
                "running. Verify with `netstat -ano | findstr :PORT` or tasklist."
                % (p.returncode, secs))
    if secs < 2:
        return "(no output, exit %s, %.1fs)" % (p.returncode, secs)
    return "(no output, exit %s)" % p.returncode


def tool_append(args, cwd):
    """Append to a file (or create it). This is how a long file gets built without
    ever emitting one huge tool argument."""
    raw = (args.get("path") or "").strip()
    content = args.get("content", "")
    if not raw:
        return "error: path is required"
    if not content:
        return "error: no content supplied, nothing was appended"
    p = _resolve(raw, cwd)
    if os.path.isdir(p):
        return "error: %s is a DIRECTORY, not a file" % raw
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    except OSError as e:
        return f"error: {e}"
    last = None
    for attempt in range(4):
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
            return ("appended %d chars to %s (now %d bytes)"
                    % (len(content), raw, os.path.getsize(p)))
        except PermissionError as e:
            last = e
            time.sleep(0.4 * (attempt + 1))
        except OSError as e:
            return f"error: {e}"
    return ("error: %s is locked by another process (%s) - write to a different "
            "filename." % (raw, last))


def _walk(root, globpat="", skip=None):
    """Yield candidate files under root. Deterministic order, skip heavy dirs."""
    skip = set(d.lower() for d in (skip or []))
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in skip)
        for fn in sorted(filenames):
            if globpat and not fnmatch.fnmatch(fn.lower(), globpat.lower()):
                continue
            yield os.path.join(dirpath, fn)


def tool_grep(args, cwd):
    """path:line: text - the cheap way to look inside files."""
    pat = args.get("pattern") or ""
    if not pat:
        return "error: pattern is required"
    try:
        rx = re.compile(pat)
    except re.error as e:
        return "error: bad regex: %s" % e
    cfg = CFG["tools"]["grep"]
    root = _resolve(args.get("path") or ".", cwd)
    if not os.path.exists(root):
        return "error: no such path: %s" % (args.get("path") or ".")
    ctx = max(0, min(5, int(args.get("context", cfg["context"]) or 0)))
    maxm = max(1, min(200, int(args.get("max") or cfg["max_matches"])))
    maxline = int(cfg["max_line_chars"])
    hits, files_hit, skipped_big = [], 0, 0
    for fp in _walk(root, args.get("glob") or "", cfg["skip_dirs"]):
        try:
            if os.path.getsize(fp) > int(cfg["max_file_bytes"]):
                skipped_big += 1
                continue
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        rel = os.path.relpath(fp, cwd) if os.path.isdir(cwd) else fp
        rel = rel.replace("\\", "/")
        found_here = False
        shown = set()
        for no, line in enumerate(lines):
            if not rx.search(line):
                continue
            found_here = True
            lo, hi = max(0, no - ctx), min(len(lines), no + ctx + 1)
            for k in range(lo, hi):
                if k in shown:          # overlapping context must not repeat lines
                    continue
                shown.add(k)
                mark = ":" if k == no else "-"
                txt = lines[k][:maxline]
                hits.append("%s%s%d%s %s" % (rel, mark, k + 1, mark, txt))
            if len(hits) >= maxm:
                break
        if found_here:
            files_hit += 1
        if len(hits) >= maxm:
            break
    if not hits:
        return "no matches for %r under %s%s" % (
            pat, args.get("path") or ".", " (glob %s)" % args["glob"] if args.get("glob") else "")
    more = " (stopped at %d matches - raise max= or narrow the glob)" % maxm if len(hits) >= maxm else ""
    big = " (%d files skipped: larger than the limit)" % skipped_big if skipped_big else ""
    return "%d match lines in %d file(s)%s%s\n%s" % (
        len(hits), files_hit, more, big, "\n".join(hits))


def tool_list(args, cwd):
    """Directory listing with sizes - replaces `dir /b` and `ls -la` guesswork."""
    root = _resolve(args.get("path") or ".", cwd)
    if not os.path.exists(root):
        return "error: no such path: %s" % (args.get("path") or ".")
    globpat = args.get("glob") or "*"
    rec = bool(args.get("recursive"))
    if os.path.isfile(root):
        st = os.stat(root)
        return "%s  %8d B  %s" % (args.get("path"), st.st_size,
                                  time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)))
    rows, cap = [], 250
    if rec:
        for fp in _walk(root, globpat):
            st = os.stat(fp)
            rows.append("%-58s %9d B  %s" % (os.path.relpath(fp, root)[:58], st.st_size,
                                             time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))))
            if len(rows) >= cap:
                rows.append("... (stopped at %d entries)" % cap)
                break
    else:
        for name in sorted(os.listdir(root)):
            fp = os.path.join(root, name)
            if not fnmatch.fnmatch(name.lower(), globpat.lower()):
                continue
            try:
                st = os.stat(fp)
            except OSError:
                continue
            kind = "DIR " if os.path.isdir(fp) else "    "
            rows.append("%s%-54s %9d B  %s" % (kind, name[:54], st.st_size,
                                               time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))))
            if len(rows) >= cap:
                rows.append("... (stopped at %d entries)" % cap)
                break
    head = "%s (%d entries%s)\n" % (args.get("path") or ".", len(rows),
                                    ", recursive" if rec else "")
    return head + ("\n".join(rows) if rows else "(empty)")


DISPATCH = {
    "read": tool_read,
    "grep": tool_grep,
    "list": tool_list,
    "append": tool_append,
    "write": tool_write,
    "edit": tool_edit,
    "bash": tool_bash,
}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
# Degenerate-output detector - SHARED with llm-certify.py so the agent and the
# certifier cannot drift apart. They once did: the certifier printed CLEAN while
# the answer ended "mem0-!!!!!!!!" because it required 12+ repeats and only knew
# about "@". Rules live in poison_detect.py; do not fork them again.
def _load_poison_detect():
    """Shared detector (poison_detect.py, same rules as llm-certify). Degrades to
    "never poisoned" if the file is missing, but never invents a CLEAN verdict
    while crashing either."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        from poison_detect import analyze
        return analyze
    except Exception:
        pass
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "poison_detect", os.path.join(here, "poison_detect.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.analyze
    except Exception:
        return lambda text: []


_analyze_poison = _load_poison_detect()


def _poisoned(r):
    """True when the runner returned a poisoned (degenerate) response.

    A poisoned llama-server answers HTTP 200 with finish_reason=stop, empty
    content, and a wall of ONE repeated character in the reasoning channel.
    Cause: the Gated-DeltaNet recurrent state on the GPU is garbage. It is
    process-resident - a new prompt or a slot erase does NOT clear it, only a
    server restart does.
    """
    return bool(_analyze_poison((r.get("content") or "") + (r.get("reasoning") or "")))


def _restart_server(server, wait_s=150):
    """Restart the model server via llm-stop.ps1 + llm-load.ps1.

    Returns True only when the server answers /v1/models again. This is the
    ONLY thing that clears a poisoned runner, so the agent uses it to heal
    itself instead of surfacing a broken run to the operator.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    stop = os.path.join(here, "llm-stop.ps1")
    load = os.path.join(here, "llm-load.ps1")
    if not (os.path.isfile(stop) and os.path.isfile(load)):
        return False
    ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    try:
        subprocess.run(ps + [stop], capture_output=True, timeout=150)
        subprocess.run(ps + [load], capture_output=True, timeout=240)
    except Exception:
        return False
    base = server.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    for _ in range(max(1, wait_s // 3)):
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=4):
                return True
        except Exception:
            time.sleep(3)
    return False


def server_alive(server):
    """Is the model server answering at all? Used to tell "it was stopped"
    apart from "it is up but returned garbage" - different operator action."""
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/v1/models", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def served_alias(server, timeout=4):
    """The alias the runner answers with (/v1/models), or "".
    Used to fail fast when the loaded weights are not the configured model."""
    try:
        base = server.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as r:
            d = json.load(r)
        ms = d.get("models") or d.get("data") or []
        if ms:
            return str(ms[0].get("name") or ms[0].get("id") or "").strip()
    except Exception:
        pass
    return ""


def call_model(server, model, messages, tools, max_tokens, temperature, on_delta):
    """Stream one completion. on_delta(kind, text) fires per token chunk."""
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        # llama-server reports the REAL prompt size in a final chunk when asked.
        # Without it every token count in the UI is a chars/3.6 guess.
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "presence_penalty": 0,
        # 1.05, not 1.1: the higher value warps structured output (tool-call
        # JSON, code) and we no longer need it as a repetition safety net - the
        # @-wall was never repetition-driven (it was the GDN state, fixed by QF0).
        "repeat_penalty": 1.05,
        "top_p": 0.9,
        "top_k": 20,
    }
    req = urllib.request.Request(
        f"{server}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    calls = {}
    content = []
    reasoning = []
    usage = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            s = line[5:].strip()
            if s == "[DONE]":
                break
            try:
                ev = json.loads(s)
            except Exception:
                continue
            # the usage chunk arrives with "choices": [] right before [DONE]
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                rsn = d.get("reasoning_content") or d.get("reasoning")
                if rsn:
                    reasoning.append(rsn)
                    on_delta("reasoning", rsn)
                if d.get("content"):
                    content.append(d["content"])
                    on_delta("content", d["content"])
                for tc in d.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    idx = tc.get("index", 0)
                    e = calls.setdefault(idx, {"id": None, "name": "", "args": ""})
                    if tc.get("id"):
                        e["id"] = tc["id"]
                    if fn.get("name"):
                        e["name"] = fn["name"]
                    if fn.get("arguments"):
                        e["args"] += fn["arguments"]
                        on_delta("toolargs", fn["arguments"])
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "calls": [calls[k] for k in sorted(calls)],
        "usage": usage,
        "seconds": time.time() - t0,
    }


def parse_args_json(raw):
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


# --------------------------------------------------------------------------
# compaction
# --------------------------------------------------------------------------
def _digest_message(m):
    """One line of useful signal from a message (no noise)."""
    role = m.get("role")
    if role == "user":
        c = str(m.get("content") or "")
        if c.startswith("[COMPACTED SUMMARY"):
            return "USER: (earlier summary)"
        return f"USER: {c[:300]}"
    if role == "assistant":
        tcs = m.get("tool_calls") or []
        c = str(m.get("content") or "")
        if tcs:
            names = ", ".join((t.get("function") or {}).get("name", "?") for t in tcs)
            if c:
                return f"ASSISTANT (called {names}): {c[:300]}"
            return f"ASSISTANT called {names}"
        return f"ASSISTANT: {c[:400]}" if c else "ASSISTANT: (no text)"
    if role == "tool":
        out = str(m.get("content") or "").strip()
        lines = [x for x in out.splitlines() if x.strip()]
        if not lines:
            return "TOOL: (no output)"
        head = lines[0][:160]
        tail = lines[-1][:160]
        return f"TOOL: {head}" + (f" ... {tail}" if len(lines) > 2 else "")
    return ""


def session_lock_path(sid):
    return session_path(sid) + ".lock"


def pid_alive(pid):
    """True if a Windows process with this id exists and has not exited."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                        # STILL_ACTIVE
    except Exception:
        return False


def acquire_session_lock(sid, stale_after=3 * 3600):
    """A lock file next to the session. The old pair raced: COMPACT spawned a
    second process while the run kept rewriting sess["messages"] after every
    step, so the compaction was silently lost.

    A forced STOP (taskkill /F) can never run the atexit release, so a dead
    holder's PID is treated as a stale lock immediately instead of blocking the
    session for stale_after seconds."""
    p = session_lock_path(sid)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)   # first run: dir may not exist yet
        if os.path.exists(p):
            age = time.time() - os.path.getmtime(p)
            holder = 0
            try:
                holder = int((open(p, encoding="utf-8").read().strip() or "0"))
            except (OSError, ValueError):
                holder = 0
            if holder and not pid_alive(holder):
                os.remove(p)                            # dead holder: take it over
            elif age < stale_after:
                return False, "session %s is in use by another process (pid %s, lock %ds old)" % (
                    sid, holder or "?", int(age))
            else:
                os.remove(p)
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True, ""
    except OSError as e:
        return False, "could not lock the session: %s" % e


def release_session_lock(sid):
    try:
        os.remove(session_lock_path(sid))
    except OSError:
        pass


def trim_tool_results(messages, keep_last=6, chars=1200):
    """Tool-result clearing: the cheapest compaction (Anthropic's own advice).

    No model call, no information about WHAT was done is lost - only the bulk of
    old tool payloads. Frees the tokens without paying 60 s of summarisation.
    Returns chars saved.
    """
    saved, seen = 0, 0
    for m in reversed(messages):
        if m.get("role") != "tool":
            continue
        seen += 1
        if seen <= max(0, keep_last // 2):
            continue
        c = m.get("content") or ""
        if len(c) <= chars:
            continue
        m["content"] = c[:chars] + ("\n...[old tool result trimmed to %d chars to save "
                                    "context; the full output is in the transcript]" % chars)
        saved += len(c) - len(m["content"])
    return saved


def compact_session(sess, server, model, emit, keep_last=None):
    """Shrink a long session: summarise the middle, keep the head and the tail.

    The LLM summary is best-effort: this model spends its whole token budget
    thinking and can return an EMPTY answer (finish_reason=length). When that
    happens we fall back to a deterministic digest that keeps the useful signal
    (task, tool names, last line of each tool result, final answers).
    """
    ccfg = CFG["compaction"]
    keep_last = int(keep_last or ccfg["keep_last_messages"])
    keep_tokens = int(ccfg["keep_last_tokens"])
    ok, why = acquire_session_lock(sess.get("id") or "")
    if not ok:
        emit({"type": "compact", "changed": False, "reason": why})
        return False
    try:
        return _compact_locked(sess, server, model, emit, keep_last, ccfg)
    finally:
        release_session_lock(sess.get("id") or "")


def _compact_locked(sess, server, model, emit, keep_last, ccfg):
    keep_tokens = int(ccfg["keep_last_tokens"])
    msgs = sess.get("messages") or []
    before_tokens = estimate_context(msgs)

    if len(msgs) <= keep_last + 2:
        emit({"type": "compact", "changed": False,
              "reason": f"only {len(msgs)} messages, nothing to compact"})
        return False

    head = msgs[:2]           # system + original task
    # keep_last MESSAGES is the wrong unit: six 5 000-char tool results are ~4 k
    # tokens. Take from the end until either budget runs out.
    tail, tail_tokens = [], 0
    for m in reversed(msgs[2:]):
        t = estimate_tokens(m.get("content") or "")
        if len(tail) >= keep_last or (tail and tail_tokens + t > keep_tokens):
            break
        tail.insert(0, m)
        tail_tokens += t
    tail = tail or msgs[-1:]
    middle = msgs[2:len(msgs) - len(tail)]

    digest = "\n".join(x for x in (_digest_message(m) for m in middle) if x)
    if not digest.strip():
        emit({"type": "compact", "changed": False, "reason": "nothing worth compacting"})
        return False

    dig_cap = int(ccfg["digest_chars"])
    if len(digest) > dig_cap:
        digest = digest[:dig_cap // 2] + "\n...\n" + digest[-dig_cap // 2:]

    # ---- best-effort LLM summary ----
    summary = ""
    try:
        req = urllib.request.Request(
            f"{server}/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content":
                        "You compress a work log into terse bullet points. "
                        "Keep file names, commands, decisions and results. "
                        "Output ONLY the bullet points, no preamble, no analysis."},
                    {"role": "user", "content":
                        "Compress this log into at most 12 bullet points:\n\n" + digest},
                ],
                "stream": False,
                "max_tokens": int(ccfg["summary_max_tokens"]),   # thinking needs room
                "temperature": 0.3,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        summary = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    except Exception as e:
        emit({"type": "compact_note", "note": f"LLM summary failed ({e}); using the digest"})

    if not summary.strip():
        summary = ("Work log so far:\n" + digest)
        source = "digest"
    else:
        source = "model"

    new_msgs = head + [{
        "role": "user",
        "content": "[COMPACTED SUMMARY OF EARLIER STEPS]\n" + summary.strip(),
    }] + tail
    after_tokens = estimate_context(new_msgs)

    if after_tokens >= before_tokens:
        emit({"type": "compact", "changed": False,
              "reason": f"no saving: {before_tokens} -> {after_tokens} tokens"})
        return False

    sess["messages"] = new_msgs
    # the transcript is deliberately left alone - it is the full history
    sess["compacted"] = True
    emit({"type": "compact", "changed": True, "source": source,
          "before": len(msgs), "after": len(new_msgs),
          "tokens_before": before_tokens, "tokens": after_tokens,
          "saved": before_tokens - after_tokens,
          "summary": summary.strip()[:2000]})
    return True


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", help="the task for the agent")
    ap.add_argument("--cwd", default="")
    ap.add_argument("--session", default="", help="session id to continue")
    ap.add_argument("--config", default=CONFIG_PATH, help="config.json for this run")
    ap.add_argument("--server", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="reply budget ceiling; clamped per step by the room left in the "
                         "context. Too small truncates a big tool call and llama-server 500s")
    ap.add_argument("--max-tool-chars", type=int, default=None,
                    help="cap on ONE tool result (chars); the model sees a head+tail clip")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="emit JSON events")
    ap.add_argument("--quiet", action="store_true", help="do not stream the model text")
    ap.add_argument("--reset", action="store_true", help="wipe the session history")
    ap.add_argument("--compact", action="store_true", help="compact the session history")
    ap.add_argument("--list-sessions", action="store_true")
    ap.add_argument("--ctx-size", type=int, default=None, help="server context window")
    args = ap.parse_args()

    # ---- config: file first, explicit flags win -------------------------
    global CFG
    CFG = config_lib.load(args.config)
    CFG["ui"]["sessions_dir"] = CFG["ui"]["sessions_dir"]
    globals()["STATE_DIR"] = CFG["ui"]["sessions_dir"]
    a = CFG["agent"]
    args.server = args.server or a["server"]
    args.model = args.model or a["model"]
    args.ctx_size = int(args.ctx_size or a["ctx_size"])
    args.max_tokens = int(args.max_tokens or a["max_tokens"])
    args.max_steps = int(args.max_steps or a["max_steps"])
    args.temperature = float(args.temperature if args.temperature is not None else a["temperature"])
    args.max_tool_chars = int(args.max_tool_chars or CFG["tools"]["max_result_chars"])
    globals()["MAX_TOOL_CHARS"] = args.max_tool_chars

    if args.list_sessions:
        for s in list_sessions():
            print(f"{s['id']:24s} {s['messages']:3d} msgs  ~{s['tokens']:5d} tok  {s['title'][:50]}")
        return

    events = []

    def emit(ev):
        events.append(ev)
        if args.json:
            print(json.dumps(ev, ensure_ascii=False), flush=True)

    task = " ".join(args.task).strip()

    # ---- session reset ---------------------------------------------------
    if args.reset:
        if not args.session:
            print("--reset needs --session", file=sys.stderr)
            sys.exit(2)
        ok, why = acquire_session_lock(args.session)
        if not ok:
            emit({"type": "error", "message": why})
            emit({"type": "done", "reason": f"FAILED - {why}", "failed": True})
            return
        try:
            sess = load_session(args.session)
            sess["messages"] = []
            sess["compacted"] = False
            sess["title"] = ""
            save_session(sess)
        finally:
            release_session_lock(args.session)
        emit({"type": "reset", "session": args.session})
        if not task:
            return

    # ---- compact-only ----------------------------------------------------
    if args.compact and not task:
        if not args.session:
            print("--compact needs --session", file=sys.stderr)
            sys.exit(2)
        sess = load_session(args.session)
        compact_session(sess, args.server, args.model, emit)
        save_session(sess)
        return

    if not task:
        ap.error("a task is required")

    # ---- normal run ------------------------------------------------------
    cwd = os.path.abspath(args.cwd) if args.cwd else os.getcwd()
    if not os.path.isdir(cwd):
        sys.exit(f"error: not a directory: {cwd}")

    if args.session:
        got_lock, why = acquire_session_lock(args.session)
        RUN_LOCKS.add(args.session)
        if not got_lock:
            emit({"type": "error", "message": why,
                  "hint": "another task or a COMPACT is using this session - wait for it to finish"})
            emit({"type": "done", "reason": f"FAILED - {why}", "failed": True})
            return
        sess = load_session(args.session)
    else:
        sess = blank_session("")
    sess["cwd"] = cwd
    if not sess.get("title"):
        sess["title"] = task[:60]

    messages = repair_dangling_tools(list(sess.get("messages") or []))
    transcript = list(sess.get("transcript") or [])
    resumed = len(messages)
    if not messages:
        messages = [{"role": "system", "content": load_system()}]
    messages.append({"role": "user", "content": f"Working directory: {cwd}\n\nTask: {task}"})
    transcript.append({"kind": "user", "text": task, "cwd": cwd})

    emit({"type": "start", "cwd": cwd, "model": args.model, "server": args.server,
          "session": sess["id"], "resumed": resumed, "ctx_size": args.ctx_size,
          "tokens_before": estimate_context(messages)})

    # persist the task immediately: even if the very first model call dies,
    # the session exists with the task in it
    if args.session:
        sess["messages"] = messages
        save_session(sess)

    # Fail fast when the runner serves different weights than the config asks
    # for: llama-server rejects a model id it does not serve, so every step
    # would die with an HTTP 400 halfway through the task. Nothing used to
    # check this - the page said MODEL UP as soon as the port answered.
    served = served_alias(args.server)
    if served and args.model and served.strip().lower() != args.model.strip().lower():
        msg = ("the model server is serving '%s' but this run asks for '%s' - the "
               "selected profile and the loaded weights disagree. Press RESTART "
               "(or set agent.model to '%s') and run again."
               % (served, args.model, served))
        emit({"type": "error", "message": msg,
              "hint": "nothing was sent to the model, so nothing was lost"})
        emit({"type": "done", "reason": "FAILED - model mismatch", "failed": True})
        return

    out_mode = "none" if args.json else ("none" if args.quiet else "text")

    def on_delta(kind, text):
        if out_mode == "text":
            if kind in ("reasoning", "content"):
                print(text, end="", flush=True)
        elif args.json:
            emit({"type": "delta", "kind": kind, "text": text})

    fatal = None
    said_done = False
    tool_json_retried = False
    compacted_this_run = False
    # real_prompt: prompt_tokens the server reported for the last call.
    # measured_est: what the estimator said at that same moment. The difference
    # between the two is the estimator's bias on THIS content, which is how the
    # meter stays exact between calls instead of drifting.
    real_prompt = 0
    measured_est = 0
    for step in range(1, args.max_steps + 1):
      try:
        # Never ask for more tokens than the context can still hold: a reply that
        # runs out mid-tool-call is what makes llama-server answer 500 (measured:
        # a 17 kB HTML page cut at 4072 of 4096 tokens).
        used_est = estimate_context(messages)
        # exact when we have a measurement, estimated only for the part added since
        used = used_est if not real_prompt else real_prompt + max(0, used_est - measured_est)
        room = args.ctx_size - used - 256
        step_tokens = min(args.max_tokens, max(512, room))
        # AUTO-COMPACT: grow.room guard. Without this a long run simply walks into
        # the ceiling and the step fails; the page could only compact BETWEEN runs.
        ccfg = CFG["compaction"]
        if (ccfg["auto"] and not compacted_this_run and step > 1
                and used > args.ctx_size * float(ccfg["trigger_pct"])):
            compacted_this_run = True
            emit({"type": "note", "message": "context at %d/%d tokens (%.0f%%) - compacting "
                                             "the middle before this step"
                  % (used, args.ctx_size, 100.0 * used / args.ctx_size)})
            if ccfg["trim_tool_results_first"]:
                saved = trim_tool_results(messages, ccfg["keep_last_messages"],
                                          ccfg["trim_tool_result_chars"])
                if saved:
                    emit({"type": "note", "message": "tool results trimmed: %d chars freed "
                                                     "(cheapest compaction, no model call)" % saved})
            if estimate_context(messages) > args.ctx_size * float(ccfg["trigger_pct"]):
                sess["messages"] = messages
                compact_session(sess, args.server, args.model, emit)
                messages = sess["messages"]
            used_est = estimate_context(messages)
            used = used_est if not real_prompt else real_prompt + max(0, used_est - measured_est)
            measured_est = used_est
            room = args.ctx_size - used - 256
            step_tokens = min(args.max_tokens, max(512, room))
        # The page needs a per-TURN boundary: without it the live reasoning of step 2
        # was appended into the step-1 THINKING box (reload looked right because it
        # rebuilds from the stored transcript).
        emit({"type": "turn", "step": step, "tokens": used, "ctx": args.ctx_size})
        call_est = used_est
        r = call_model(args.server, args.model, messages, TOOLS,
                       step_tokens, args.temperature, on_delta)
        u = r.get("usage") or {}
        if int(u.get("prompt_tokens") or 0) > 0:
            real_prompt = int(u["prompt_tokens"])
            measured_est = call_est
            usage_now = {"prompt": real_prompt,
                         "completion": int(u.get("completion_tokens") or 0),
                         "estimate": call_est, "ctx": args.ctx_size,
                         "at": time.time()}
            sess["last_usage"] = usage_now
            # the page draws the meter from THIS, not from a chars/3.6 guess
            emit({"type": "tokens", "step": step, "prompt": real_prompt,
                  "completion": usage_now["completion"], "estimate": call_est,
                  "ctx": args.ctx_size})
            transcript.append({"kind": "usage", "step": step,
                               "prompt": real_prompt, "completion": usage_now["completion"]})
      except Exception as e:
        # The model server died / was unreachable / timed out. Report it
        # instead of dying with a bare traceback, and keep what we have.
        fatal = f"{type(e).__name__}: {e}"
        # Three different failures hide behind one exception line, so classify
        # them instead of guessing from a port probe:
        #   HTTP code    -> the server ANSWERED (it is up; nothing was lost)
        #   no code+dead -> transport died: stopped/unloaded underneath us
        #   no code+up   -> request failed on a live server (poison candidate)
        code = getattr(e, "code", None)
        detail = ""
        if code:
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
        cut_off = ("Failed to parse tool call" in detail) or ("parse_error" in detail)
        if cut_off and not tool_json_retried:
            # Measured: the model emitted a whole 7 kB HTML page inside one tool
            # call and max_tokens cut the JSON mid-string; llama-server answers
            # 500 for that. One retry with an explicit "write in pieces" nudge
            # usually succeeds and costs nothing.
            tool_json_retried = True
            emit({"type": "note",
                  "message": "tool call rejected: its JSON was cut off mid-string "
                             "(too much in one call) - retrying with a write-in-chunks "
                             "instruction"})
            messages.append({"role": "user", "content":
                "Your last tool call was rejected by the server: the JSON arguments were "
                "cut off before the end because the content was too large for one reply. "
                "Do it in SECTIONS: call write with the first ~60 lines, then call append "
                "with the next ~60 lines, and repeat. Give a real FILE path (never a "
                "folder name) and never send an empty content field."})
            continue
        if code:
            hint = ("the model server answered HTTP %s - it is UP, so this is not a crash "
                    "and nothing was corrupted (no restart needed). %s"
                    % (code, ("Its tool-call JSON was cut off: ask for a smaller file or a "
                              "chunked write." if cut_off else
                              "Re-run the task; if it repeats, use RESTART for a clean runner.")))
        elif not server_alive(args.server):
            hint = ("the model server was stopped or unloaded while this task was running "
                    "(the port is closed). That is not poisoning and nothing was corrupted - "
                    "load the model again and re-run the task.")
        else:
            hint = ("the model server is up but the request failed. If the output looked "
                    "degenerate, use CHECK then RESTART - a fresh runner is the only cure.")
        transcript.append({"kind": "error", "step": step, "message": fatal, "hint": hint})
        emit({"type": "error", "message": fatal, "hint": hint})
        break
      if True:
        if out_mode == "text" and (r["content"] or r["reasoning"]):
            print("\n", flush=True)

        assistant = {"role": "assistant", "content": r["content"] or None}
        # Persist the reasoning too: a reloaded session must show the THINKING
        # blocks again, not just the final answers.
        if r["reasoning"]:
            assistant["reasoning"] = r["reasoning"]
        # The rich transcript is the UI's source of truth: it keeps the
        # reasoning, the tool CALL ARGS and the tool OUTPUT even after a
        # compaction has replaced the model-facing messages.
        transcript.append({
            "kind": "assistant",
            "step": step,
            "reasoning": r["reasoning"],
            "content": r["content"],
            "tool_calls": [{"name": c["name"], "arguments": c["args"]} for c in r["calls"]],
        })
        if r["calls"]:
            assistant["tool_calls"] = [
                {
                    "id": c["id"] or f"call_{step}_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["args"]},
                }
                for i, c in enumerate(r["calls"])
            ]
        messages.append(assistant)

        emit({
            "type": "assistant",
            "step": step,
            "seconds": round(r["seconds"], 1),
            "reasoning": r["reasoning"],
            "reasoning_len": len(r["reasoning"]),
            "content": r["content"],
            "tool_calls": [c["name"] for c in r["calls"]],
        })

        if not r["calls"]:
            # ---- POISON GUARD -------------------------------------------------
            # A poisoned llama-server (hybrid/GDN prompt-cache bug) answers
            # HTTP 200 with finish_reason=stop, content=null and a wall of ONE
            # repeated character in the reasoning channel. The model is not
            # finished - it is unable to emit tokens. Without this check the
            # agent reports a clean success that produced nothing, which is how
            # this failure stayed invisible.
            blob = (r["content"] or "") + (r["reasoning"] or "")
            reasons = _analyze_poison(blob)
            if not (r["content"] or "").strip() and reasons:
                msg = ("model runner is poisoned (degenerate output: "
                       + ", ".join(reasons)
                       + ") - only a model-server RESTART clears it")
                emit({"type": "error", "message": msg,
                      "hint": "run llm.cmd restart (a cache or slot clear does NOT help)"})
                said_done = True
                emit({"type": "done", "reason": f"FAILED - {msg}", "step": step,
                      "failed": True})
                break
            said_done = True
            emit({"type": "done", "reason": "no more tool calls", "step": step,
                  "failed": False})
            break

        # Save BEFORE the tools run. A bash call that blocks (the model started
        # a server in the foreground) used to leave the session at the previous
        # step, so a stopped run looked like it had died with no error at all.
        if args.session:
            sess["messages"] = messages
            sess["transcript"] = transcript
            save_session(sess)

        results = []
        for c in r["calls"]:
            # A tool that raises must not kill the run: one bad path (a file that
            # vanished, a permission error, a folder the OS refuses) used to end
            # the whole task with a traceback and a "the agent crashed" banner.
            try:
                fn = DISPATCH.get(c["name"])
                a = parse_args_json(c["args"])
                out = f"error: unknown tool {c['name']}" if fn is None else fn(a, cwd)
            except Exception as e:
                a = {}
                out = "error: %s: %s" % (type(e).__name__, e)
            if not isinstance(out, str):
                out = str(out)
            # Per-tool clip shape: logs and grep output matter at the END, file
            # listings at the START. One ratio for everything was wrong both ways.
            ratio = {"bash": float(CFG["tools"]["bash_tail_ratio"]), "grep": 0.15,
                     "list": 0.25, "read": 0.4}.get(c["name"], 0.5)
            out, out_chars = _clip_tool(out, args.max_tool_chars, ratio)
            results.append({"id": c["id"] or "", "name": c["name"], "output": out})
            transcript.append({"kind": "tool", "step": step, "name": c["name"],
                               "args": a, "output": out, "chars": out_chars})
            emit({"type": "tool", "step": step, "name": c["name"], "args": a,
                  "output": out[:4000], "chars": out_chars,
                  "clipped": out_chars > len(out)})
            if not args.json:
                print(f"  -> {c['name']}({json.dumps(a)[:120]})\n     {out[:300]}\n", flush=True)

        joined = "\n".join(f"[{x['name']}]\n{x['output']}" for x in results)
        # Several calls in ONE step could still sum past the budget, so cap the
        # combined payload too (3x the per-result cap).
        step_cap = int(args.max_tool_chars * float(CFG["tools"]["step_cap_multiplier"]))
        if len(joined) > step_cap:
            joined, _ = _clip_tool(joined, step_cap)
        messages.append({"role": "tool", "content": joined})

        # Persist after every step. If the run is interrupted (the browser
        # closes, the model hangs, the process is killed), the conversation so
        # far is still on disk instead of being lost.
        if args.session:
            sess["messages"] = messages
            sess["transcript"] = transcript
            save_session(sess)
    else:
        if not said_done:
            said_done = True
            emit({"type": "done", "reason": "max steps reached", "failed": False})

    if fatal and not said_done:
        said_done = True
        emit({"type": "done", "reason": f"FAILED - {fatal}", "failed": True})

    # ---- persist ---------------------------------------------------------
    if args.session:
        sess["messages"] = messages
        sess["transcript"] = transcript
        save_session(sess)
        tok = estimate_context(messages)
        shown = real_prompt or tok
        emit({"type": "saved", "session": sess["id"], "messages": len(messages),
              "tokens": shown, "estimate": tok, "exact": bool(real_prompt),
              "ctx": args.ctx_size, "usage": round(shown / args.ctx_size, 3)})

    if args.session:
        release_session_lock(args.session)

    if not args.json and out_mode == "text":
        print("\n--- final ---")
        last = messages[-1]
        if last.get("role") == "assistant" and last.get("content"):
            print(last["content"])


if __name__ == "__main__":
    try:
        main()
    finally:
        # a crashed run must not leave the session locked forever
        for _s in list(RUN_LOCKS):
            release_session_lock(_s)
