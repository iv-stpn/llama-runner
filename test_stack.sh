#!/usr/bin/env bash
# Full end-to-end test for the llama-server + LiteLLM stack (start.sh).
#
# Validates the two things we fixed:
#   1. Requests actually return (no indefinite hang) — small, streaming, and a
#      large-prompt prefill-stress request that reproduces the original symptom.
#   2. The stack shuts down quickly and cleanly, without hitting the 90s
#      llama-server force-kill and without leftover processes.
#
# It boots the stack from scratch, exercises it, then sends SIGTERM to start.sh
# and checks the teardown. start.sh traps EXIT/INT/TERM through the SAME
# cleanup(), so SIGTERM exercises the exact path a foreground Ctrl+C runs — and
# unlike SIGINT it's trappable on a backgrounded child (see phase 3 for why).
# Safe by design: it never SIGKILLs llama-server itself (that is the WSL2
# GPU-wedge risk), it only drives start.sh.
#
# Usage:   ./test_stack.sh
# Env overrides:
#   READY_TIMEOUT   (600)  max seconds to wait for the proxy to come up
#   REQ_TIMEOUT     (120)  per-request timeout for the small/streaming requests
#   BIG_REQ_TIMEOUT (240)  timeout for the large-prompt request (the hang test)
#   BIG_WORDS       (4000) size of the large prompt, in words (~1.3 tokens/word)
#   SHUTDOWN_TIMEOUT(120)  max seconds to wait for start.sh to exit after SIGTERM
#   PROXY_PORT (8000) / LLAMA_PORT (8080)
# Note: the stack runs with --no-mmap (required for stable inference on this rig
# — mmap+-cmoe crashes) plus --no-mmproj. With the vision projector gone,
# teardown completes gracefully in ~4s (measured), so phase 3 expects a clean
# exit. A force-kill is tolerated only if it lands within LLAMA_SHUTDOWN_GRACE_SEC
# (default 20s, the safety cap); phase 3 fails if the grace window is blown past
# or the process never exits.

set -uo pipefail   # deliberately NOT -e: a test runner must survive failing checks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_PORT="${PROXY_PORT:-8000}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
REQ_TIMEOUT="${REQ_TIMEOUT:-120}"
BIG_REQ_TIMEOUT="${BIG_REQ_TIMEOUT:-240}"
BIG_WORDS="${BIG_WORDS:-4000}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-120}"
START_LOG="$SCRIPT_DIR/logs/test_start.log"

START_PID=""
PYHELPER=""
GPU_BASELINE_FREE=""

# ----- output helpers -------------------------------------------------------
if [ -t 1 ]; then C_G=$'\e[32m'; C_R=$'\e[31m'; C_Y=$'\e[33m'; C_B=$'\e[1m'; C_0=$'\e[0m'
else C_G=""; C_R=""; C_Y=""; C_B=""; C_0=""; fi

PASS=0; FAIL=0
declare -a FAILED=()
section(){ printf '\n%s=== %s ===%s\n' "$C_B" "$*" "$C_0"; }
info(){ printf '   %s\n' "$*"; }
ok(){ PASS=$((PASS+1)); printf '   %s[PASS]%s %s\n' "$C_G" "$C_0" "$*"; }
bad(){ FAIL=$((FAIL+1)); FAILED+=("$*"); printf '   %s[FAIL]%s %s\n' "$C_R" "$C_0" "$*"; }
warn(){ printf '   %s[WARN]%s %s\n' "$C_Y" "$C_0" "$*"; }

# TCP connect test — version-independent readiness/liveness check.
port_open(){ (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&-; return 0; }; return 1; }
gpu_free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

# ----- guaranteed teardown --------------------------------------------------
finish(){
    local rc=$?
    if [ -n "$START_PID" ] && kill -0 "$START_PID" 2>/dev/null; then
        # SIGTERM, not SIGINT: start.sh traps EXIT INT TERM -> same cleanup(),
        # but a backgrounded child has SIGINT set to SIG_IGN (untrappable), so
        # only TERM reliably fires its teardown. Never kill -9 llama-server here
        # (WSL2 GPU-wedge risk) — let start.sh's own trap unload the model.
        warn "Test exiting while stack still up — sending SIGTERM to start.sh (PID $START_PID)..."
        kill -TERM "$START_PID" 2>/dev/null || true
        local w=0
        while kill -0 "$START_PID" 2>/dev/null && [ "$w" -lt "$SHUTDOWN_TIMEOUT" ]; do sleep 1; w=$((w+1)); done
        if kill -0 "$START_PID" 2>/dev/null; then
            warn "start.sh (PID $START_PID) is still running. Check it manually — do NOT 'kill -9' llama-server (WSL2 GPU-wedge risk); let start.sh's own trap finish."
        fi
    fi
    [ -n "$PYHELPER" ] && rm -f "$PYHELPER"
    return $rc
}
trap finish EXIT

# ----- request helper (Anthropic Messages API, stream or non-stream) --------
write_helper(){
    PYHELPER="$(mktemp "${TMPDIR:-/tmp}/req_helper.XXXXXX.py")"
    cat >"$PYHELPER" <<'PY'
import os, sys, json, time, socket, urllib.request

url = os.environ["REQ_URL"].rstrip("/") + "/v1/messages"
mode = os.environ.get("REQ_MODE", "nostream")
maxtok = int(os.environ.get("REQ_MAXTOK", "64"))
timeout = float(os.environ.get("REQ_TIMEOUT", "120"))
big = int(os.environ.get("REQ_BIG_WORDS", "0"))
prompt = os.environ.get("REQ_PROMPT", "")

if big > 0:
    filler = ("The quick brown fox jumps over the lazy dog. " * (big // 9 + 1)).split()[:big]
    prompt = ("Reference text (ignore it): " + " ".join(filler) +
              "\n\nIgnore everything above. Reply with exactly one word: PONG")

body = {"model": "local", "max_tokens": maxtok,
        "messages": [{"role": "user", "content": prompt}]}
if mode == "stream":
    body["stream"] = True

req = urllib.request.Request(
    url, data=json.dumps(body).encode(), method="POST",
    headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01",
             "content-type": "application/json"})

t0 = time.monotonic(); ttft = None
try:
    resp = urllib.request.urlopen(req, timeout=timeout)
except socket.timeout:
    print("FAIL no-response-within-%.0fs (hang: no bytes during prefill)" % timeout); sys.exit(1)
except Exception as e:
    print("FAIL request-error: %r" % e); sys.exit(1)

def extract_text(ev):
    d = ev.get("delta") or {}
    if isinstance(d, dict) and (d.get("text") or d.get("thinking")):
        return d.get("text") or d.get("thinking") or ""
    for ch in ev.get("choices") or []:            # tolerate OpenAI-shaped chunks
        cd = ch.get("delta") or {}
        if cd.get("content"): return cd["content"]
    return ""

if mode == "stream":
    events = 0; chars = 0; done = False
    try:
        for raw in resp:
            if ttft is None and raw.strip():
                ttft = time.monotonic() - t0
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload in ("[DONE]", ""):
                done = True; continue
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            events += 1
            chars += len(extract_text(ev))
            if ev.get("type") in ("message_stop",) or (ev.get("choices") or [{}])[0].get("finish_reason"):
                done = True
            if time.monotonic() - t0 > timeout:
                print("FAIL stream-exceeded-%.0fs" % timeout); sys.exit(1)
    except socket.timeout:
        print("FAIL stream-stall-within-%.0fs" % timeout); sys.exit(1)
    except Exception as e:
        print("FAIL stream-error: %r" % e); sys.exit(1)
    total = time.monotonic() - t0
    if events == 0:
        print("FAIL stream-no-events total=%.1fs (no SSE data — possibly an error body)" % total); sys.exit(1)
    if chars == 0:
        # Stream flowed and completed (no hang — the property under test), but
        # carried no text: the thinking-model + LiteLLM reasoning-drop quirk.
        # WARN (exit 2), not FAIL.
        print("NOTE streamed total=%.1fs ttft=%.1fs events=%d but 0 text chars — "
              "thinking-only / proxy drops reasoning" %
              (total, ttft if ttft is not None else -1, events)); sys.exit(2)
    print("OK ttft=%.1fs total=%.1fs events=%d chars=%d done=%s" %
          (ttft if ttft is not None else -1, total, events, chars, done)); sys.exit(0)
else:
    try:
        raw = resp.read()
    except socket.timeout:
        print("FAIL read-within-%.0fs (hang)" % timeout); sys.exit(1)
    total = time.monotonic() - t0
    try:
        obj = json.loads(raw.decode("utf-8", "replace"))
    except Exception as e:
        print("FAIL bad-json: %r body=%.160r" % (e, raw[:160])); sys.exit(1)
    if isinstance(obj, dict) and obj.get("type") == "error":
        print("FAIL api-error: %.240r" % obj); sys.exit(1)
    stop = obj.get("stop_reason") if isinstance(obj, dict) else None
    content = obj.get("content", []) if isinstance(obj, dict) else []
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    think = "".join(b.get("thinking", "") for b in content if b.get("type") == "thinking")
    if not (text.strip() or think.strip()):
        # Bounded, valid, non-error response => the stack did NOT hang (the
        # property under test). Empty content is the known thinking-model +
        # LiteLLM quirk: the proxy drops reasoning_content, and at a small token
        # budget the whole budget can be spent on reasoning. Report as a NOTE
        # (exit 2 => WARN), not a hang failure.
        print("NOTE responded total=%.1fs but no text block (stop_reason=%r) — "
              "thinking-only / proxy drops reasoning" % (total, stop)); sys.exit(2)
    print("OK total=%.1fs textchars=%d thinkchars=%d stop=%s answer=%r" %
          (total, len(text), len(think), stop, text.strip()[:70] or "(thinking-only)")); sys.exit(0)
PY
}

run_req(){   # run_req "<label>" KEY=VAL KEY=VAL ...
    local label="$1"; shift
    info "-> $label"
    local out rc
    out="$(env "$@" REQ_URL="http://localhost:$PROXY_PORT" python3 "$PYHELPER" 2>&1)"; rc=$?
    # 0 = got text (pass), 2 = responded promptly but thinking-only/no text
    # (WARN: the stack did NOT hang, which is what phase 2 tests), 1 = real
    # failure (hang, transport error, api error).
    case "$rc" in
        0) ok   "$label  [$out]" ;;
        2) warn "$label  [$out]" ;;
        *) bad  "$label  [$out]" ;;
    esac
}

# ===========================================================================
printf '%sllama-server + LiteLLM stack — end-to-end test%s\n' "$C_B" "$C_0"
info "repo: $SCRIPT_DIR   proxy: :$PROXY_PORT   backend: :$LLAMA_PORT"

# ----- 0. Preflight ---------------------------------------------------------
section "0. Preflight"
[ -x "$SCRIPT_DIR/start.sh" ] || { bad "start.sh not found or not executable"; exit 1; }
if [ ! -x "$SCRIPT_DIR/llama.cpp/llama-server" ]; then
    bad "llama.cpp/llama-server missing — start.sh would try to BUILD it (needs sudo, will block on redirected stdin). Build first with: ./start.sh --rebuild"
    exit 1
fi
if port_open "$PROXY_PORT" || port_open "$LLAMA_PORT"; then
    bad "port $PROXY_PORT or $LLAMA_PORT already in use — a stack is likely already running. Stop it first."
    exit 1
fi
if ! find "$SCRIPT_DIR/unsloth" -name '*.gguf' -print -quit 2>/dev/null | grep -q .; then
    warn "model not cached yet — start.sh will download it; READY_TIMEOUT=${READY_TIMEOUT}s may be too short."
fi
command -v python3 >/dev/null 2>&1 || { bad "python3 not found (required for request tests)"; exit 1; }
GPU_BASELINE_FREE="$(gpu_free_mb)"
[ -n "$GPU_BASELINE_FREE" ] && info "GPU free at rest: ${GPU_BASELINE_FREE} MB"
write_helper
ok "preflight clean (binary present, ports free, python3 available)"

# ----- 1. Startup -----------------------------------------------------------
section "1. Startup"
: > "$START_LOG"
info "launching $SCRIPT_DIR/start.sh (output -> $START_LOG)"
"$SCRIPT_DIR/start.sh" >"$START_LOG" 2>&1 &
START_PID=$!
info "start.sh PID=$START_PID; waiting up to ${READY_TIMEOUT}s for proxy on :$PROXY_PORT ..."
t0=$(date +%s); last=0; ready=0
while :; do
    if ! kill -0 "$START_PID" 2>/dev/null; then
        bad "start.sh exited during startup (see log below)"
        info "---- last 30 lines of $START_LOG ----"; tail -n 30 "$START_LOG"
        exit 1
    fi
    if port_open "$PROXY_PORT"; then ready=1; break; fi
    el=$(( $(date +%s) - t0 ))
    if [ "$el" -ge "$READY_TIMEOUT" ]; then
        bad "proxy not ready within ${READY_TIMEOUT}s"
        info "---- last 30 lines of $START_LOG ----"; tail -n 30 "$START_LOG"
        exit 1
    fi
    if [ $((el - last)) -ge 15 ]; then info "... still waiting (${el}s elapsed)"; last=$el; fi
    sleep 2
done
ok "proxy reachable on :$PROXY_PORT after $(( $(date +%s) - t0 ))s"

# ----- 2. Backend + request responsiveness ---------------------------------
section "2. Responsiveness (the 'hangs indefinitely' fix)"
if command -v curl >/dev/null 2>&1; then
    if curl -sf --max-time 5 "http://localhost:$LLAMA_PORT/health" >/dev/null 2>&1; then
        ok "llama-server /health (:$LLAMA_PORT) responding"
    else
        bad "llama-server /health (:$LLAMA_PORT) not responding"
    fi
fi
# Token budgets must be generous: Qwen3.6 is a thinking model that spends a few
# hundred tokens in reasoning_content before any answer text, and LiteLLM 1.90.2
# drops reasoning instead of mapping it to a `thinking` block. At max_tokens=64
# the whole budget is consumed by (dropped) reasoning and content arrives empty
# — which is a proxy/model quirk, NOT the hang under test. 2048 reliably yields
# a clean end_turn with real text (verified: "The capital of France is Paris.").
run_req "small non-streaming request"     REQ_MODE=nostream REQ_MAXTOK=2048 REQ_TIMEOUT="$REQ_TIMEOUT" \
        REQ_PROMPT="What is the capital of France? Answer in one short sentence."
run_req "streaming request (Claude Code path)" REQ_MODE=stream REQ_MAXTOK=2048 REQ_TIMEOUT="$REQ_TIMEOUT" \
        REQ_PROMPT="Count from 1 to 5, one number per line."
# The large-prompt test's real metric is TTFT: proving prefill of a big prompt
# doesn't stall (the original symptom). 1024 tokens leaves room to think AND
# emit text after the ~5k-token prefill, still well inside BIG_REQ_TIMEOUT.
run_req "large-prompt prefill (~${BIG_WORDS} words, was the hang)" REQ_MODE=stream REQ_MAXTOK=1024 \
        REQ_BIG_WORDS="$BIG_WORDS" REQ_TIMEOUT="$BIG_REQ_TIMEOUT"

# ----- 3. Shutdown (teardown path / Ctrl+C) --------------------------------
section "3. Shutdown / teardown (the 'won't stop on Ctrl+C' fix)"
# We send SIGTERM, not SIGINT, and that is deliberate. start.sh traps
# `EXIT INT TERM` -> the SAME cleanup() function, so TERM exercises the exact
# teardown a foreground Ctrl+C runs. We CAN'T use SIGINT here: bash sets SIGINT
# to SIG_IGN for a child it backgrounds in a non-interactive script, and a
# signal ignored at entry can't be trapped, so start.sh's INT trap never arms
# when launched with '&' (verified: SIGINT produced no "Shutting down"). The
# user's real `./start.sh` runs in the foreground of an interactive shell where
# SIGINT is NOT ignored, so their Ctrl+C fires this identical cleanup path.
info "sending SIGTERM to start.sh (PID $START_PID) — fires the same cleanup() as a foreground Ctrl+C ..."
sd0=$(date +%s)
kill -TERM "$START_PID" 2>/dev/null
while kill -0 "$START_PID" 2>/dev/null; do
    el=$(( $(date +%s) - sd0 ))
    [ "$el" -ge "$SHUTDOWN_TIMEOUT" ] && break
    sleep 1
done
if kill -0 "$START_PID" 2>/dev/null; then
    bad "start.sh did not exit within ${SHUTDOWN_TIMEOUT}s of SIGTERM"
else
    wait "$START_PID" 2>/dev/null; ec=$?
    sd=$(( $(date +%s) - sd0 ))
    info "start.sh exited after ${sd}s (exit code $ec)"
    # The property under test is "start.sh exits promptly and frees the GPU."
    # With --no-mmap + --no-mmproj, teardown completes gracefully in ~4s (the
    # 90s hang was the BF16 vision projector, now dropped), so a clean exit is
    # the expected path. A force-kill is still tolerated as long as it lands
    # within the grace cap — but if it fires at all, something re-wedged the
    # graceful unload and it's worth a look, so we note it rather than staying
    # silent. Real failures: blowing past the grace window, or not exiting
    # (caught above).
    grace="${LLAMA_SHUTDOWN_GRACE_SEC:-20}"
    if grep -q "forcing termination" "$START_LOG"; then
        if [ "$sd" -le $(( grace + 15 )) ]; then
            ok "shutdown in ${sd}s via force-kill after ~${grace}s grace (graceful unload didn't finish — check if unexpected; GPU reclaim verified next)"
        else
            bad "shutdown took ${sd}s — ${grace}s force-kill grace not honored / teardown wedged"
        fi
    elif [ "$sd" -le 30 ]; then
        ok "clean graceful shutdown in ${sd}s (no force-kill needed)"
    else
        ok "shutdown clean (no force-kill) but slow: ${sd}s"
    fi
fi

# ----- 4. Post-shutdown state ----------------------------------------------
section "4. Post-shutdown state"
sleep 2
if pgrep -f "$SCRIPT_DIR/llama.cpp/llama-server" >/dev/null 2>&1; then bad "leftover llama-server process"; else ok "no leftover llama-server process"; fi
if pgrep -f "$SCRIPT_DIR/.venv/bin/litellm"     >/dev/null 2>&1; then bad "leftover litellm process";     else ok "no leftover litellm process"; fi
if port_open "$PROXY_PORT"; then bad "port $PROXY_PORT still listening"; else ok "port $PROXY_PORT released"; fi
if port_open "$LLAMA_PORT"; then bad "port $LLAMA_PORT still listening"; else ok "port $LLAMA_PORT released"; fi
if [ -n "$GPU_BASELINE_FREE" ]; then
    nowfree="$(gpu_free_mb)"
    info "GPU free: baseline ${GPU_BASELINE_FREE} MB -> now ${nowfree:-?} MB"
    if [ -n "$nowfree" ] && [ "$nowfree" -ge $(( GPU_BASELINE_FREE - 300 )) ]; then
        ok "GPU memory released back to ~baseline"
    else
        warn "GPU free below baseline (WSL2 accounting can lag a few seconds)"
    fi
fi

# ----- Summary --------------------------------------------------------------
section "Summary"
printf '   passed: %s%d%s   failed: %s%d%s\n' "$C_G" "$PASS" "$C_0" "$([ "$FAIL" -gt 0 ] && echo "$C_R" || echo "$C_G")" "$FAIL" "$C_0"
if [ "$FAIL" -gt 0 ]; then
    printf '%s   failing checks:%s\n' "$C_R" "$C_0"
    for f in "${FAILED[@]}"; do printf '     - %s\n' "$f"; done
    info "full start.sh output: $START_LOG"
    exit 1
fi
printf '   %sAll checks passed — stack starts, serves, and shuts down cleanly.%s\n' "$C_G" "$C_0"
exit 0
