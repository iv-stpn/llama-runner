#!/usr/bin/env bash
# Start llama-server + LiteLLM proxy for Claude Code.
# LiteLLM translates Anthropic Messages API → OpenAI → llama-server.
# Usage: ./start.sh [--rebuild]
#   --rebuild  force-update and rebuild llama.cpp before starting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
VENV_DIR="$SCRIPT_DIR/.venv"
LLAMA_PORT=8080
PROXY_PORT=8000
MODEL_REPO="unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
MODEL_FILE="UD-Q3_K_XL"

# Known-good llama.cpp commit verified on this rig (2026-07-06). Pinning
# keeps rebuilds reproducible; override with LLAMA_CPP_REF=master to chase
# upstream (e.g. to check whether the draft-mtp crash #23210 got fixed).
LLAMA_CPP_REF="${LLAMA_CPP_REF:-f36e5c348bc8795c34f9a038e58876e7a8423d4d}"

# OOM guard settings (override via env if needed).
# The KV-cache estimate is intentionally conservative for large MoE models.
LLAMA_CTX_TARGET="${LLAMA_CTX_TARGET:-256000}"
LLAMA_CTX_MIN="${LLAMA_CTX_MIN:-8192}"
OOM_RESERVED_GB="${OOM_RESERVED_GB:-3}"
KV_BYTES_PER_TOKEN_ESTIMATE="${KV_BYTES_PER_TOKEN_ESTIMATE:-360000}"

# Runtime guard settings: terminate gracefully before WSL gets OOM-killed.
RUNTIME_OOM_GUARD_ENABLE="${RUNTIME_OOM_GUARD_ENABLE:-1}"
RUNTIME_CHECK_INTERVAL_SEC="${RUNTIME_CHECK_INTERVAL_SEC:-2}"
RUNTIME_MIN_FREE_GB="${RUNTIME_MIN_FREE_GB:-2}"
RUNTIME_LOW_STREAK_LIMIT="${RUNTIME_LOW_STREAK_LIMIT:-5}"
RUNTIME_WRITEBACK_WARN_MB="${RUNTIME_WRITEBACK_WARN_MB:-512}"

# Graceful-shutdown budget before SIGKILL. The original 90s hang was --no-mmap
# (kept — mmap+-cmoe crashes inference, see launch args) TOGETHER with the
# auto-loaded BF16 vision projector: freeing that projector through WSL2's GPU
# passthrough is what stalled teardown. With --no-mmproj the projector is gone
# and teardown now completes on its own in ~4s (measured 2026-07-06), never
# reaching this cap. So this is a safety net, not the fix: if some future change
# re-wedges teardown, force-kill after 20s instead of the old 90s of dead time
# (SIGKILL freed the GPU cleanly every time it hit today).
LLAMA_SHUTDOWN_GRACE_SEC="${LLAMA_SHUTDOWN_GRACE_SEC:-20}"
# Under MEMORY PRESSURE a graceful SIGTERM is actively harmful: llama-server's
# exit handler (the "cleaning up before exit..." line) walks 25GB of resident
# weights to free them, and doing that while the box is already near-OOM thrashes
# swap so hard the teardown appears to hang — which is exactly the reported
# symptom. So the memory-pressure guard uses a much shorter grace before SIGKILL:
# reclaiming the pages wholesale (what SIGKILL does) is what actually relieves the
# pressure. Empirically SIGKILL has freed the GPU cleanly on this rig every time.
LLAMA_GUARD_KILL_GRACE_SEC="${LLAMA_GUARD_KILL_GRACE_SEC:-3}"
# After SIGKILL, how long to wait for the kernel to actually reap the process
# before declaring it unreapable. A process wedged in an uninterruptible GPU
# ioctl (WSL2 /dev/dxg) can't be reaped even by SIGKILL until that syscall
# returns; past this window we stop waiting and tell the operator to run
# `wsl.exe --shutdown` rather than polling a zombie forever.
LLAMA_SIGKILL_REAP_WAIT_SEC="${LLAMA_SIGKILL_REAP_WAIT_SEC:-10}"

# Extra stability guards for long WSL runs.
# The low-VRAM guard arms only after the server passes its health check and
# uses the steady-state free VRAM at that moment as its baseline: on small
# GPUs a fully loaded model legitimately leaves only a few hundred MB free,
# so a fixed floor would kill a healthy server (this happened — see git log).
RUNTIME_MAX_GPU_TEMP_C="${RUNTIME_MAX_GPU_TEMP_C:-86}"
RUNTIME_GPU_MIN_FREE_MB="${RUNTIME_GPU_MIN_FREE_MB:-512}"
RUNTIME_GPU_LOW_STREAK_LIMIT="${RUNTIME_GPU_LOW_STREAK_LIMIT:-4}"

# Conservative serving defaults to avoid transient memory spikes during
# long-context decoding. At/above the WSL threshold below these get clamped
# further; override via env if you want to experiment at small context.
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-256}"
LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-64}"
# MTP speculative decoding is off by default: the draft-mtp path hits CUDA
# illegal memory access under multi-turn agentic load (llama.cpp #23210).
# Set LLAMA_SPEC_DRAFT_N_MAX=1 to re-enable once fixed upstream.
LLAMA_SPEC_DRAFT_N_MAX="${LLAMA_SPEC_DRAFT_N_MAX:-0}"
# At/above this context on WSL, "WSL safety mode" (see configure_runtime_safety)
# turns Flash Attention OFF, disables the draft path, and clamps the batch.
# This is load-bearing, NOT theoretical: with FA ON at ctx=64000 this rig
# reproduced the CUDA illegal-memory-access crash (2026-07-06, draft-mtp already
# off), and a crashed backend looks exactly like an indefinite hang to the
# client. Must sit below 64000: choose_safe_ctx halves from 256000 and lands on
# 64000, never 65536, so a 65536 threshold would silently skip safety mode.
LLAMA_DISABLE_FA_CTX_THRESHOLD="${LLAMA_DISABLE_FA_CTX_THRESHOLD:-49152}"

# Auto-restart supervision. When llama-server dies UNEXPECTEDLY (CUDA illegal
# memory access, segfault, external kill) the supervisor relaunches it instead
# of tearing the whole stack down — LiteLLM stays up so Claude Code keeps its
# endpoint and just sees a short window of connection errors during reload.
# A resource GUARD stop is restarted only when killing llama-server itself clears
# the condition: memory-pressure and gpu-low-vram are SELF-CLEARING (llama IS the
# ~24GB RSS / the VRAM tenant, so a fresh generation lands in a clean state) and
# restart by default under the usual streak cap. GPU overtemp is NOT self-clearing
# (the die needs real cooldown, not a 5s retry) and stays terminal unless
# LLAMA_RESTART_ON_GUARD_STOP=1. A *-wedged passthrough is ALWAYS terminal (only
# `wsl.exe --shutdown` clears it) regardless of that opt-in.
LLAMA_RESTART_ENABLE="${LLAMA_RESTART_ENABLE:-1}"
# Max consecutive failed generations before giving up. A generation that stays
# healthy for LLAMA_HEALTHY_RESET_SEC resets this counter, so only a *rapid*
# crash loop (e.g. a deterministic load-time crash) ever exhausts it.
LLAMA_MAX_RESTARTS="${LLAMA_MAX_RESTARTS:-5}"
# Backoff between restarts: base doubles each consecutive failure up to a cap.
# The 5s base is deliberate — after a CUDA abort the driver/context needs a
# moment to settle before a fresh process can allocate cleanly on WSL2.
LLAMA_RESTART_BACKOFF_BASE_SEC="${LLAMA_RESTART_BACKOFF_BASE_SEC:-5}"
LLAMA_RESTART_BACKOFF_MAX_SEC="${LLAMA_RESTART_BACKOFF_MAX_SEC:-60}"
# A generation must run healthy at least this long to count as "recovered" and
# clear the consecutive-failure counter.
LLAMA_HEALTHY_RESET_SEC="${LLAMA_HEALTHY_RESET_SEC:-120}"
# Opt-in: also restart after a GPU-overtemp stop (the only remaining terminal-by-
# default guard; memory-pressure and low-VRAM already restart). *-wedged ignores
# this — a stuck passthrough can't be retried away.
LLAMA_RESTART_ON_GUARD_STOP="${LLAMA_RESTART_ON_GUARD_STOP:-0}"

# Persistent telemetry for post-mortem debugging after abrupt crashes/reboots.
GUARD_LOG_ENABLE="${GUARD_LOG_ENABLE:-1}"
GUARD_LOG_FILE="${GUARD_LOG_FILE:-$SCRIPT_DIR/logs/llama_guard.log}"
GUARD_HEARTBEAT_SEC="${GUARD_HEARTBEAT_SEC:-30}"
GUARD_STATE_FILE="${GUARD_STATE_FILE:-$SCRIPT_DIR/logs/llama_guard.state}"
# llama-server output is captured here (rotated to .prev each run) so CUDA
# errors survive the terminal scrollback and abrupt session ends.
LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-$SCRIPT_DIR/logs/llama_server.log}"
# Mirror that captured log to the console so the live per-task tok/s and
# slot-processing lines (llama-server's verbose output) show up in the terminal
# again, not just in the file. On by default; set 0 for a quiet console (the
# file is still written either way). This is a decoupled `tail -f` reader, NOT
# a pipe on llama-server's stdout: piping (tee) would let a Ctrl+C on the reader
# push SIGPIPE into llama-server mid-CUDA-teardown — the exact GPU-wedge the
# setsid isolation below exists to prevent.
LLAMA_LOG_TO_CONSOLE="${LLAMA_LOG_TO_CONSOLE:-1}"
# The watchdog runs in a subshell: its variable assignments never reach the
# cleanup trap in the main script, so stop reasons are handed over via file.
GUARD_REASON_FILE="${GUARD_REASON_FILE:-$SCRIPT_DIR/logs/llama_guard.reason}"
# Touched once the health check passes; tells the watchdog the server is
# fully loaded and the low-VRAM guard may arm.
READY_FLAG_FILE="$SCRIPT_DIR/logs/llama_ready.flag"

GUARD_EXIT_REASON="normal-exit"

log_guard() {
    [ "$GUARD_LOG_ENABLE" = "1" ] || return 0
    local ts msg
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    msg="$*"
    mkdir -p "$(dirname "$GUARD_LOG_FILE")"
    echo "[$ts] $msg" >> "$GUARD_LOG_FILE"
}

write_guard_state() {
    mkdir -p "$(dirname "$GUARD_STATE_FILE")"
    cat > "$GUARD_STATE_FILE" <<EOF
start_ts=$(date '+%Y-%m-%d %H:%M:%S')
script_pid=$$
reason=$GUARD_EXIT_REASON
EOF
}

remove_guard_state() {
    rm -f "$GUARD_STATE_FILE"
}

report_runtime_stop() {
    # Called from the watchdog subshell so the main script's cleanup trap
    # can log the real stop reason instead of "normal-exit".
    mkdir -p "$(dirname "$GUARD_REASON_FILE")"
    echo "$1" > "$GUARD_REASON_FILE"
}

check_unclean_previous_run() {
    [ -f "$GUARD_STATE_FILE" ] || return 0
    local prev_start prev_reason
    prev_start=$(awk -F= '/^start_ts=/{print substr($0,10)}' "$GUARD_STATE_FILE" 2>/dev/null || true)
    prev_reason=$(awk -F= '/^reason=/{print substr($0,8)}' "$GUARD_STATE_FILE" 2>/dev/null || true)
    echo "==> Detected previous unclean stop (possible crash/reboot)."
    [ -n "$prev_start" ] && echo "==> Previous run started at: $prev_start"
    [ -n "$prev_reason" ] && echo "==> Last recorded reason: $prev_reason"
    log_guard "unclean_previous_run prev_start=${prev_start:-unknown} prev_reason=${prev_reason:-unknown}"
}

install_cuda_toolkit() {
    echo "==> Installing CUDA toolkit for WSL2..."
    local DEB="cuda-keyring_1.1-1_all.deb"
    curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/$DEB" -o "/tmp/$DEB"
    sudo dpkg -i "/tmp/$DEB"
    # Debian 13 rejects NVIDIA's SHA1-bound GPG key via sqv — bypass for this repo only.
    sudo apt-get update -qq -o Acquire::AllowInsecureRepositories=true 2>/dev/null || true
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated cuda-toolkit-12-8
    rm -f "/tmp/$DEB"
}

build_llama() {
    echo "==> Installing build dependencies..."
    # AllowInsecureRepositories tolerates the NVIDIA repo's SHA1 key on Debian 13
    sudo apt-get update -qq -o Acquire::AllowInsecureRepositories=true 2>/dev/null || true
    sudo apt-get install -y pciutils build-essential cmake curl libcurl4-openssl-dev

    # Detect CUDA: need nvcc (compiler), not just nvidia-smi (driver).
    local CUDA_FLAG=OFF
    if command -v nvcc &>/dev/null; then
        CUDA_FLAG=ON
        echo "==> CUDA toolkit found: $(nvcc --version | head -1)"
    elif command -v nvidia-smi &>/dev/null; then
        echo "==> NVIDIA GPU detected but CUDA toolkit missing — installing..."
        install_cuda_toolkit
        export PATH="/usr/local/cuda/bin:$PATH"
        CUDA_FLAG=ON
    else
        echo "==> No NVIDIA GPU detected, building CPU-only (slow inference)."
    fi

    if [ -d "$LLAMA_DIR/.git" ]; then
        echo "==> Updating existing llama.cpp checkout..."
        git -C "$LLAMA_DIR" fetch origin
    else
        echo "==> Cloning llama.cpp..."
        rm -rf "$LLAMA_DIR"
        git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    fi
    # origin/<ref> first so branch names track upstream, not a stale local.
    echo "==> Checking out llama.cpp ref: $LLAMA_CPP_REF"
    git -C "$LLAMA_DIR" checkout --detach "origin/$LLAMA_CPP_REF" 2>/dev/null \
        || git -C "$LLAMA_DIR" checkout --detach "$LLAMA_CPP_REF"

    echo "==> Building llama.cpp (GGML_CUDA=$CUDA_FLAG)..."
    cmake "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
        -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA="$CUDA_FLAG"
    # Cap parallelism: unbounded -j spawns one nvcc/cicc per core, each using
    # 1-2GB+ RAM compiling CUDA kernels, which can OOM-kill the build.
    # ~2GB RAM per job is a safe rule of thumb.
    local TOTAL_RAM_GB
    TOTAL_RAM_GB=$(awk '/^MemTotal:/ {printf "%d", $2/1024/1024; exit}' /proc/meminfo)
    local BUILD_JOBS=$(( TOTAL_RAM_GB / 2 ))
    [ "$BUILD_JOBS" -lt 1 ] && BUILD_JOBS=1
    echo "==> Building with -j$BUILD_JOBS (detected ${TOTAL_RAM_GB}GB RAM)..."
    cmake --build "$LLAMA_DIR/build" --config Release -j"$BUILD_JOBS" --clean-first \
        --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
    cp "$LLAMA_DIR/build/bin/llama-"* "$LLAMA_DIR/"
    echo "==> llama.cpp build complete."
}

if [ "${1:-}" = "--rebuild" ] || [ ! -f "$LLAMA_DIR/llama-server" ]; then
    build_llama
fi

if [ ! -x "$VENV_DIR/bin/litellm" ]; then
    echo "==> Setting up Python venv for LiteLLM proxy..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

LLAMA_MATCH="^$LLAMA_DIR/llama-server"

kill_stale_llama_server() {
    local pids
    pids=$(pgrep -f "$LLAMA_MATCH" || true)
    [ -z "$pids" ] && return
    echo "==> Found a leftover llama-server process from a previous run — stopping it..."
    kill $pids 2>/dev/null || true
    local waited=0
    while pgrep -f "$LLAMA_MATCH" >/dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [ $((waited % 10)) -eq 0 ]; then
            echo "==> Still waiting for the leftover process to release GPU memory (${waited}s)..."
        fi
        if [ "$waited" -ge 90 ]; then
            echo "==> Leftover llama-server didn't exit after 90s — force killing..."
            echo "==> If loading now hangs, WSL2's GPU passthrough may be wedged;"
            echo "==> run 'wsl.exe --shutdown' from PowerShell (not a full reboot) and try again."
            pkill -9 -f "$LLAMA_MATCH" 2>/dev/null || true
            sleep 1
            break
        fi
    done
}
kill_stale_llama_server

kill_stale_litellm() {
    # A leftover LiteLLM keeps port $PROXY_PORT bound, so the fresh instance
    # dies on "address already in use" while llama-server looks perfectly
    # healthy — the proxy is then unreachable even though the server runs.
    local pids waited=0
    pids=$(pgrep -f "$VENV_DIR/bin/litellm" || true)
    [ -z "$pids" ] && return
    echo "==> Found a leftover LiteLLM proxy from a previous run — stopping it to free port $PROXY_PORT..."
    kill $pids 2>/dev/null || true
    while pgrep -f "$VENV_DIR/bin/litellm" >/dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge 10 ]; then
            pkill -9 -f "$VENV_DIR/bin/litellm" 2>/dev/null || true
            sleep 1
            break
        fi
    done
}
kill_stale_litellm

choose_safe_ctx() {
    local target_ctx="$LLAMA_CTX_TARGET"
    local min_ctx="$LLAMA_CTX_MIN"

    local mem_total_kb mem_avail_kb swap_free_kb usable_kb reserve_kb
    mem_total_kb=$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo)
    mem_avail_kb=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)
    swap_free_kb=$(awk '/^SwapFree:/ {print $2; exit}' /proc/meminfo)
    reserve_kb=$(( OOM_RESERVED_GB * 1024 * 1024 ))

    # Use currently available memory + free swap, minus a safety reserve.
    usable_kb=$(( mem_avail_kb + swap_free_kb - reserve_kb ))
    if [ "$usable_kb" -lt $((2 * 1024 * 1024)) ]; then
        usable_kb=$((2 * 1024 * 1024))
    fi

    local model_bytes model_kb
    model_bytes=$(find "$LLAMA_CACHE" -name '*.gguf' -printf '%s\n' 2>/dev/null | sort -nr | head -1)
    if [ -z "$model_bytes" ]; then
        # Conservative fallback when model is not cached yet.
        model_kb=$((22 * 1024 * 1024))
    else
        model_kb=$(( model_bytes / 1024 ))
    fi

    local overhead_kb kv_per_token_kb required_kb ctx
    overhead_kb=$((2 * 1024 * 1024))
    kv_per_token_kb=$(( KV_BYTES_PER_TOKEN_ESTIMATE / 1024 ))
    ctx="$target_ctx"

    # Try to keep requested context, then degrade gracefully in powers of two.
    while true; do
        required_kb=$(( (model_kb * 12 / 10) + overhead_kb + (ctx * kv_per_token_kb) ))
        if [ "$required_kb" -le "$usable_kb" ]; then
            break
        fi
        if [ "$ctx" -le "$min_ctx" ]; then
            ctx="$min_ctx"
            break
        fi
        ctx=$(( ctx / 2 ))
        if [ "$ctx" -lt "$min_ctx" ]; then
            ctx="$min_ctx"
        fi
    done

    required_kb=$(( (model_kb * 12 / 10) + overhead_kb + (ctx * kv_per_token_kb) ))
    if [ "$required_kb" -gt "$usable_kb" ]; then
        echo "==> Not enough available memory to start safely (needs ~$(awk "BEGIN {printf \"%.1f\", $required_kb/1024/1024}")GB, has ~$(awk "BEGIN {printf \"%.1f\", $usable_kb/1024/1024}")GB after reserve)." >&2
        echo "==> Refusing to start to avoid WSL OOM. Free RAM, add swap, or set a smaller LLAMA_CTX_TARGET." >&2
        GUARD_EXIT_REASON="startup-memory-precheck-failed"
        log_guard "startup_precheck_failed need_gb=$(awk "BEGIN {printf \"%.2f\", $required_kb/1024/1024}") have_gb=$(awk "BEGIN {printf \"%.2f\", $usable_kb/1024/1024}") ctx=$ctx"
        return 1
    fi

    if [ "$ctx" -lt "$target_ctx" ]; then
        echo "==> Memory guard lowered context from $target_ctx to $ctx to avoid OOM."
        log_guard "startup_ctx_lowered target_ctx=$target_ctx chosen_ctx=$ctx"
    fi

    LLAMA_CTX="$ctx"
    log_guard "startup_ctx_selected ctx=$LLAMA_CTX"
    return 0
}

is_wsl() {
    grep -qiE 'microsoft|wsl' /proc/sys/kernel/osrelease 2>/dev/null
}

configure_runtime_safety() {
    # Base defaults for small-context / non-WSL use, where FA is a clear win.
    LLAMA_FA_MODE="on"
    LLAMA_SPEC_ENABLED=1

    # WSL safety mode. On WSL at large context this rig hits a CUDA illegal
    # memory access; FA-ON at ctx=64000 REPRODUCED it on 2026-07-06 with the
    # draft-mtp path already disabled, so FA itself — not just draft-mtp — is
    # implicated here. A crashed backend is indistinguishable from an indefinite
    # hang at the client (LiteLLM just waits on the dead socket), so this guard
    # is the real fix for the "hangs forever" symptom, not a mere perf knob.
    # Turn FA off, disable the draft path, and clamp batch/ubatch for headroom.
    if is_wsl && [ "$LLAMA_CTX" -ge "$LLAMA_DISABLE_FA_CTX_THRESHOLD" ]; then
        LLAMA_FA_MODE="off"
        LLAMA_SPEC_DRAFT_N_MAX=0
        LLAMA_SPEC_ENABLED=0

        # Without FA the attention buffer scales with context, so keep the
        # batch small to avoid transient VRAM spikes on the 8GB card.
        if [ "$LLAMA_BATCH_SIZE" -gt 192 ]; then
            LLAMA_BATCH_SIZE=192
        fi
        if [ "$LLAMA_UBATCH_SIZE" -gt 48 ]; then
            LLAMA_UBATCH_SIZE=48
        fi

        echo "==> WSL safety mode: large context detected (ctx=$LLAMA_CTX)."
        echo "==> Applying conservative runtime flags: -fa off, no speculative draft, reduced batch/ubatch."
        log_guard "wsl_safety_mode enabled ctx=$LLAMA_CTX fa=$LLAMA_FA_MODE batch=$LLAMA_BATCH_SIZE ubatch=$LLAMA_UBATCH_SIZE"
    fi
}

echo "==> Starting llama-server on port $LLAMA_PORT..."
export LLAMA_CACHE="$SCRIPT_DIR/unsloth/Qwen3.6-35B-A3B-MTP-GGUF"

check_unclean_previous_run
write_guard_state
rm -f "$READY_FLAG_FILE" "$GUARD_REASON_FILE"

# The `-hf` downloader hits Hugging Face on every startup to check for
# updates, even once the model is fully cached — if that network call
# stalls (flaky DNS/rate limiting), the script hangs before llama-server
# ever logs a single line. Skip it entirely once we already have the model.
HF_EXTRA_ARGS=()
if find "$LLAMA_CACHE" -name '*.gguf' -print -quit 2>/dev/null | grep -q . \
    && ! find "$LLAMA_CACHE" -name '*.downloadInProgress' -print -quit 2>/dev/null | grep -q .; then
    echo "==> Model already cached — starting offline (skips the Hugging Face network check)."
    HF_EXTRA_ARGS=(--offline)
fi

choose_safe_ctx
configure_runtime_safety

LLAMA_ARGS=(
    -hf "$MODEL_REPO:$MODEL_FILE"
    --no-mmproj -cmoe -c "$LLAMA_CTX" -fa "$LLAMA_FA_MODE" -np 1 --no-mmap
    -b "$LLAMA_BATCH_SIZE" -ub "$LLAMA_UBATCH_SIZE"
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00
    --host 0.0.0.0 --port "$LLAMA_PORT"
)

if [ "$LLAMA_SPEC_ENABLED" = "1" ] && [ "$LLAMA_SPEC_DRAFT_N_MAX" -gt 0 ]; then
    LLAMA_ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$LLAMA_SPEC_DRAFT_N_MAX")
fi

if [ "${#HF_EXTRA_ARGS[@]}" -gt 0 ]; then
    LLAMA_ARGS+=("${HF_EXTRA_ARGS[@]}")
fi

LLAMA_PID=""
MONITOR_PID=""
LITELLM_PID=""
LOGTAIL_PID=""

# launch_llama_server <generation>
# setsid detaches this into its own session so a second Ctrl+C at the
# terminal doesn't land on llama-server directly. If a raw SIGINT hits it
# mid-CUDA-teardown, the driver cleanup can be left half-done, which on
# WSL2 can wedge the GPU passthrough (/dev/dxg) until `wsl --shutdown` —
# the script's trap below is the only thing that should ever signal it.
# Generation 1 uses the freshly-rotated log; restarts APPEND under a banner so
# a full crash-and-recovery sequence stays in one file for post-mortem reading
# (the .prev rotation already happened once, before the tail mirror attached).
launch_llama_server() {
    local generation="$1"
    if [ "$generation" -gt 1 ]; then
        {
            echo ""
            echo "===================================================================="
            echo "==> RESTART generation=$generation at $(date '+%Y-%m-%d %H:%M:%S')"
            echo "===================================================================="
        } >>"$LLAMA_LOG_FILE"
        setsid "$LLAMA_DIR/llama-server" "${LLAMA_ARGS[@]}" >>"$LLAMA_LOG_FILE" 2>&1 &
    else
        setsid "$LLAMA_DIR/llama-server" "${LLAMA_ARGS[@]}" >"$LLAMA_LOG_FILE" 2>&1 &
    fi

    # setsid forks internally on some systems, so $! can point to a short-lived
    # wrapper instead of the real process — find the actual PID by its command
    # line so kill/kill -0 always target the real llama-server.
    LLAMA_PID=""
    local _
    for _ in $(seq 1 100); do
        LLAMA_PID=$(pgrep -f "$LLAMA_MATCH" | head -1 || true)
        [ -n "$LLAMA_PID" ] && break
        sleep 0.1
    done
    [ -n "$LLAMA_PID" ]
}

mkdir -p "$(dirname "$LLAMA_LOG_FILE")"
[ -f "$LLAMA_LOG_FILE" ] && mv -f "$LLAMA_LOG_FILE" "$LLAMA_LOG_FILE.prev"
echo "==> llama-server output is captured in $LLAMA_LOG_FILE"

# Mirror the captured log to this terminal (see LLAMA_LOG_TO_CONSOLE above).
# -F (not -f) so it tolerates the .prev rotation and the file not existing yet;
# -n 0 starts from the launch we're about to kick off (no stale replay), and it
# survives restarts because it follows the path, not a PID. cleanup() kills it
# on exit so it can't outlive the server. It only reads the file, so a Ctrl+C on
# this tail can never reach llama-server's stdout.
if [ "$LLAMA_LOG_TO_CONSOLE" = "1" ]; then
    echo "==> Mirroring llama-server log to this console (set LLAMA_LOG_TO_CONSOLE=0 to disable)."
    tail -n 0 -F "$LLAMA_LOG_FILE" 2>/dev/null &
    LOGTAIL_PID="$!"
fi

cleanup() {
    # Capture the status that triggered this trap BEFORE anything else runs,
    # so a supervisor-chosen exit code (e.g. give-up-after-restarts) survives.
    local exit_code=$?
    # Run only once even though EXIT/INT/TERM can all fire this trap.
    trap - EXIT INT TERM
    echo "==> Shutting down..."
    # The watchdog subshell can't set GUARD_EXIT_REASON here directly — pick
    # up the reason it recorded on disk, if any.
    if [ -f "$GUARD_REASON_FILE" ]; then
        GUARD_EXIT_REASON=$(cat "$GUARD_REASON_FILE" 2>/dev/null || echo "$GUARD_EXIT_REASON")
    fi
    log_guard "cleanup begin reason=$GUARD_EXIT_REASON"
    write_guard_state
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    if [ -n "$LOGTAIL_PID" ] && kill -0 "$LOGTAIL_PID" 2>/dev/null; then
        kill "$LOGTAIL_PID" 2>/dev/null || true
    fi
    if [ -n "$LITELLM_PID" ] && kill -0 "$LITELLM_PID" 2>/dev/null; then
        kill "$LITELLM_PID" 2>/dev/null || true
    fi
    # LLAMA_PID may not be populated yet if we're interrupted while still
    # discovering it below — fall back to finding it by command line so a
    # Ctrl+C in that window doesn't leave it orphaned with nothing to kill it.
    local target="$LLAMA_PID"
    [ -z "$target" ] && target=$(pgrep -f "$LLAMA_MATCH" | head -1 || true)
    if [ -n "$target" ] && kill -0 "$target" 2>/dev/null; then
        # Graceful teardown lets llama-server free GPU memory on its own (a 35B
        # model can take a while), then terminate_llama_pid escalates to SIGKILL
        # and CONFIRMS the reap — so this can no longer poll a wedged process
        # forever the way the original open-coded loop could.
        # If a guard already diagnosed a wedge, skip the graceful grace entirely:
        # waiting on a SIGTERM a wedged process can't service is pure dead time.
        local cleanup_grace="$LLAMA_SHUTDOWN_GRACE_SEC"
        case "$GUARD_EXIT_REASON" in
            *wedged*) cleanup_grace=1 ;;
            *) echo "==> Waiting for llama-server to release GPU memory (large models can take a minute)..." ;;
        esac
        if ! terminate_llama_pid "$target" "$cleanup_grace"; then
            echo "==> llama-server did not die even after SIGKILL — WSL2's GPU passthrough is wedged."
            echo "==> Run 'wsl.exe --shutdown' from PowerShell (not a full reboot) before starting again."
            log_guard "cleanup_sigkill_unreaped pid=$target"
        fi
    fi

    rm -f "$READY_FLAG_FILE" "$GUARD_REASON_FILE"
    remove_guard_state
    log_guard "cleanup done reason=$GUARD_EXIT_REASON"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

# Terminate a llama-server PID with escalation, and CONFIRM it actually died.
#   terminate_llama_pid <pid> [grace_sec]
# One SIGTERM, wait up to grace_sec, then SIGKILL — then wait up to
# LLAMA_SIGKILL_REAP_WAIT_SEC for the kernel to actually reap it.
# Returns 0 if the process is gone, 1 if it survived even SIGKILL (wedged in an
# uninterruptible GPU ioctl — the WSL2 /dev/dxg stall, only clearable by
# `wsl.exe --shutdown`). Both the guard branches (watchdog SUBSHELL) and
# cleanup() use this: the guards can't lean on cleanup()'s escalation because the
# supervisor can't exit until the backend is dead, and the OLD code just polled
# `kill -0` forever after a bare SIGTERM — which is the reported hang, where
# llama-server printed "cleaning up before exit..." and then never returned.
terminate_llama_pid() {
    local pid="$1" grace="${2:-$LLAMA_SHUTDOWN_GRACE_SEC}" waited=0
    kill -0 "$pid" 2>/dev/null || return 0
    kill "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        [ "$waited" -ge "$grace" ] && break
    done
    kill -0 "$pid" 2>/dev/null || return 0

    # Graceful window elapsed and it's still alive — force it.
    log_guard "force_kill grace_s=$grace pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
    waited=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$LLAMA_SIGKILL_REAP_WAIT_SEC" ]; then
            # SIGKILL couldn't reap it inside the window: uninterruptible sleep.
            log_guard "sigkill_unreaped pid=$pid wait_s=$LLAMA_SIGKILL_REAP_WAIT_SEC"
            return 1
        fi
    done
    return 0
}

runtime_watchdog() {
    local target_pid="$1"
    local script_pid="$2"
    local interval="$RUNTIME_CHECK_INTERVAL_SEC"
    # Armed post-ready with an adaptive baseline (see below) — NOT a fixed floor.
    local free_floor_kb=""
    local low_streak=0
    local warned_writeback=0
    local gpu_low_streak=0
    local gpu_free_floor_mb=""
    local heartbeat_countdown="$GUARD_HEARTBEAT_SEC"

    while kill -0 "$target_pid" 2>/dev/null; do
        local mem_avail_kb swap_free_kb free_kb rss_kb dirty_kb writeback_kb
        # Anchor field names: an unanchored /Writeback/ also matches
        # "WritebackTmp:", yielding a two-line value that breaks the numeric
        # tests below (line "[: 0\n0: integer expression expected"). exit after
        # the first hit keeps each capture a single integer.
        mem_avail_kb=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)
        swap_free_kb=$(awk '/^SwapFree:/ {print $2; exit}' /proc/meminfo)
        dirty_kb=$(awk '/^Dirty:/ {print $2; exit}' /proc/meminfo)
        writeback_kb=$(awk '/^Writeback:/ {print $2; exit}' /proc/meminfo)
        free_kb=$(( mem_avail_kb + swap_free_kb ))
        rss_kb=$(awk '/^VmRSS:/ {print $2; exit}' "/proc/$target_pid/status" 2>/dev/null || echo 0)

        heartbeat_countdown=$((heartbeat_countdown - interval))
        if [ "$heartbeat_countdown" -le 0 ]; then
            log_guard "heartbeat free_gb=$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}") rss_gb=$(awk "BEGIN {printf \"%.2f\", $rss_kb/1024/1024}") writeback_mb=$(awk "BEGIN {printf \"%.0f\", $writeback_kb/1024}")"
            heartbeat_countdown="$GUARD_HEARTBEAT_SEC"
        fi

        # Arm the system-RAM guard only once the server is READY, and baseline it
        # against the free RAM at that moment — do NOT use a fixed floor. On this
        # rig a fully loaded model legitimately leaves the box near the floor
        # (llama RSS ~24GB, ~1.9GB free is the healthy steady state), so a fixed
        # 2GB floor kills a HEALTHY server the first time a big request transiently
        # bumps RSS — which is exactly what happened (served 6215s, then one 64k
        # request tipped it over for 10s and it got killed). This is the same fix
        # the GPU VRAM guard already has. If steady-state free is already below
        # 2×min, floor at half of it and only fire on a real further collapse.
        if [ -z "$free_floor_kb" ] && [ -f "$READY_FLAG_FILE" ]; then
            local min_floor_kb=$(( RUNTIME_MIN_FREE_GB * 1024 * 1024 ))
            if [ "$free_kb" -lt $(( min_floor_kb * 2 )) ]; then
                free_floor_kb=$(( free_kb / 2 ))
            else
                free_floor_kb="$min_floor_kb"
            fi
            log_guard "sys_mem_guard_armed baseline_free_kb=$free_kb floor_kb=$free_floor_kb"
        fi

        # Until armed (still loading), skip the low-memory logic entirely: free RAM
        # legitimately collapses during load and firing then is a false positive.
        if [ -n "$free_floor_kb" ] && [ "$free_kb" -lt "$free_floor_kb" ]; then
            low_streak=$((low_streak + 1))
            if [ "$low_streak" -eq 1 ]; then
                echo "==> Runtime guard warning: free RAM dropped below the post-load baseline floor (~$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")GB free incl. swap, floor ~$(awk "BEGIN {printf \"%.2f\", $free_floor_kb/1024/1024}")GB, llama RSS ~$(awk "BEGIN {printf \"%.2f\", $rss_kb/1024/1024}")GB)."
            fi
        else
            low_streak=0
            warned_writeback=0
        fi

        if [ -n "$free_floor_kb" ] && [ "$free_kb" -lt "$free_floor_kb" ] && [ "$writeback_kb" -gt $((RUNTIME_WRITEBACK_WARN_MB * 1024)) ] && [ "$warned_writeback" -eq 0 ]; then
            echo "==> Runtime guard notice: high kernel writeback (~$(awk "BEGIN {printf \"%.0f\", $writeback_kb/1024}")MB, dirty ~$(awk "BEGIN {printf \"%.0f\", $dirty_kb/1024}")MB)."
            echo "==> This can indicate cache/writeback pressure rather than pure model allocation growth."
            log_guard "writeback_pressure writeback_mb=$(awk "BEGIN {printf \"%.0f\", $writeback_kb/1024}") dirty_mb=$(awk "BEGIN {printf \"%.0f\", $dirty_kb/1024}") free_gb=$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}")"
            warned_writeback=1
        fi

        if [ "$low_streak" -ge "$RUNTIME_LOW_STREAK_LIMIT" ]; then
            echo "==> Runtime guard: memory pressure stayed critical for $((low_streak * interval))s."
            echo "==> Stopping llama-server to avoid crashing WSL."
            report_runtime_stop "runtime-system-memory-pressure"
            log_guard "runtime_stop reason=runtime-system-memory-pressure free_gb=$(awk "BEGIN {printf \"%.2f\", $free_kb/1024/1024}") rss_gb=$(awk "BEGIN {printf \"%.2f\", $rss_kb/1024/1024}")"
            # Force-kill FAST here, not a graceful SIGTERM. Under memory pressure a
            # graceful teardown is exactly the trap that hung the reported run:
            # llama-server catches SIGTERM, prints "cleaning up before exit...",
            # then tries to free ~25GB of --no-mmap weights while the box is
            # already near-OOM — every page it touches faults through swap, so the
            # teardown crawls or stalls. A SIGKILL lets the kernel reclaim the whole
            # RSS at once, which is what actually relieves the pressure. The short
            # grace still gives it a beat to drop the GPU context cleanly first.
            terminate_llama_pid "$target_pid" "$LLAMA_GUARD_KILL_GRACE_SEC" \
                || report_runtime_stop "runtime-system-memory-pressure-wedged"
            # Leave the reason on disk and exit the watchdog only. The SIGKILL just
            # freed llama's ~24GB RSS, so the pressure is already gone — the
            # supervisor reads this reason and restarts into a now-clean box (unless
            # the kill wedged, which upgrades the reason to -wedged == terminal).
            return
        fi

        if command -v nvidia-smi >/dev/null 2>&1; then
            local gpu_sample gpu_temp gpu_mem_total gpu_mem_used gpu_mem_free
            gpu_sample=$(nvidia-smi --query-gpu=temperature.gpu,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
            if [ -n "$gpu_sample" ]; then
                gpu_temp=$(echo "$gpu_sample" | awk -F',' '{gsub(/ /,"",$1); print $1}')
                gpu_mem_total=$(echo "$gpu_sample" | awk -F',' '{gsub(/ /,"",$2); print $2}')
                gpu_mem_used=$(echo "$gpu_sample" | awk -F',' '{gsub(/ /,"",$3); print $3}')
                if [ -n "$gpu_temp" ] && [ -n "$gpu_mem_total" ] && [ -n "$gpu_mem_used" ]; then
                    gpu_mem_free=$(( gpu_mem_total - gpu_mem_used ))

                    if [ "$gpu_temp" -ge "$RUNTIME_MAX_GPU_TEMP_C" ]; then
                        echo "==> Runtime guard: GPU temperature reached ${gpu_temp}C (limit ${RUNTIME_MAX_GPU_TEMP_C}C)."
                        echo "==> Stopping llama-server to prevent potential driver/system instability."
                        report_runtime_stop "runtime-gpu-overtemp"
                        log_guard "runtime_stop reason=runtime-gpu-overtemp gpu_temp_c=$gpu_temp gpu_free_mb=$gpu_mem_free"
                        terminate_llama_pid "$target_pid" \
                            || report_runtime_stop "runtime-gpu-overtemp-wedged"
                        return
                    fi

                    # Arm the low-VRAM guard only once the server reports
                    # healthy: free VRAM legitimately collapses while the
                    # model loads, and on small GPUs the loaded steady state
                    # itself sits below any sane fixed floor (8GB card ↔
                    # ~300MB free is normal). Baseline the first post-ready
                    # sample and only treat a real drop below it as danger.
                    if [ -z "$gpu_free_floor_mb" ] && [ -f "$READY_FLAG_FILE" ]; then
                        if [ "$gpu_mem_free" -lt $(( RUNTIME_GPU_MIN_FREE_MB * 2 )) ]; then
                            gpu_free_floor_mb=$(( gpu_mem_free / 2 ))
                        else
                            gpu_free_floor_mb="$RUNTIME_GPU_MIN_FREE_MB"
                        fi
                        log_guard "gpu_guard_armed baseline_free_mb=$gpu_mem_free floor_mb=$gpu_free_floor_mb"
                    fi

                    if [ -n "$gpu_free_floor_mb" ]; then
                        if [ "$gpu_mem_free" -lt "$gpu_free_floor_mb" ]; then
                            gpu_low_streak=$((gpu_low_streak + 1))
                        else
                            gpu_low_streak=0
                        fi

                        if [ "$gpu_low_streak" -ge "$RUNTIME_GPU_LOW_STREAK_LIMIT" ]; then
                            echo "==> Runtime guard: GPU free VRAM stayed below ${gpu_free_floor_mb}MB for $((gpu_low_streak * interval))s (post-load baseline floor)."
                            echo "==> Stopping llama-server to avoid GPU driver reset/system crash."
                            report_runtime_stop "runtime-gpu-low-vram"
                            log_guard "runtime_stop reason=runtime-gpu-low-vram gpu_free_mb=$gpu_mem_free floor_mb=$gpu_free_floor_mb"
                            terminate_llama_pid "$target_pid" \
                                || report_runtime_stop "runtime-gpu-low-vram-wedged"
                            return
                        fi
                    fi
                fi
            fi
        fi

        sleep "$interval"
    done

    # Reaching here means llama-server vanished. Either a guard above already
    # killed it (and recorded a reason via report_runtime_stop) or it crashed on
    # its own (e.g. CUDA illegal memory access). We do NOT tear the script down
    # here anymore — the supervisor loop notices the watchdog exit, reads the
    # reason, and decides whether to auto-restart or stop. Only record a reason
    # if a guard didn't already set one, so a guard-stop isn't misread as a crash.
    if [ ! -s "$GUARD_REASON_FILE" ]; then
        echo "==> llama-server died unexpectedly — last lines of $LLAMA_LOG_FILE:"
        tail -n 25 "$LLAMA_LOG_FILE" 2>/dev/null || true
        report_runtime_stop "llama-server-died-unexpectedly"
        log_guard "runtime_stop reason=llama-server-died-unexpectedly"
    fi
}

# Spawn a fresh watchdog bound to the current LLAMA_PID. Called once per
# generation: each restart gets a new llama-server PID, so the old watchdog
# (which polls a now-dead PID) must be reaped first — the supervisor does that
# before relaunching. Passing "$$" lets the watchdog log against this script.
start_watchdog() {
    [ "$RUNTIME_OOM_GUARD_ENABLE" = "1" ] || return 0
    runtime_watchdog "$LLAMA_PID" "$$" &
    MONITOR_PID="$!"
}

# Poll /health until the server answers or dies. Returns 0 when ready, 1 if
# llama-server exited before it ever became healthy (a load-time crash — the
# supervisor treats that specially so a deterministic startup crash can't spin
# the restart loop forever).
# llama-server's own download progress bar only prints when stdout is a tty,
# so it silently disappears when backgrounded in some shells/IDEs. Poll the
# partial download file on disk instead — this works everywhere. The curl
# timeouts matter: without them a stuck connection attempt (e.g. WSL2
# NAT hiccup) blocks this loop's condition check forever, which looks
# identical to llama-server itself being hung.
wait_for_health() {
    echo "==> Waiting for llama-server to be ready..."
    local last_status="" elapsed=0 next_heartbeat=15 status dl_file
    while ! curl -sf --connect-timeout 3 --max-time 5 "http://localhost:$LLAMA_PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "==> llama-server exited before it became healthy — last lines of $LLAMA_LOG_FILE:" >&2
            tail -n 25 "$LLAMA_LOG_FILE" >&2 2>/dev/null || true
            return 1
        fi

        dl_file=$(find "$LLAMA_CACHE" -name '*.downloadInProgress' 2>/dev/null | head -1)
        if [ -n "$dl_file" ]; then
            status="Downloading model... $(du -h "$dl_file" 2>/dev/null | cut -f1)"
        else
            status="Loading..."
        fi
        if [ "$status" != "$last_status" ]; then
            echo "==> $status"
            last_status="$status"
        elif [ "$elapsed" -ge "$next_heartbeat" ]; then
            echo "==> Still $status (${elapsed}s elapsed — large models can take several minutes)"
            next_heartbeat=$((next_heartbeat + 15))
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "==> llama-server ready."
    touch "$READY_FLAG_FILE"
    return 0
}

start_litellm() {
    echo "==> Starting LiteLLM proxy on port $PROXY_PORT..."
    log_guard "litellm_start port=$PROXY_PORT"
    "$VENV_DIR/bin/litellm" --config "$SCRIPT_DIR/litellm_config.yaml" --port "$PROXY_PORT" &
    LITELLM_PID="$!"
}

# ---------------------------------------------------------------------------
# Supervision loop.
#
# The original script launched llama-server exactly once, and the watchdog
# tore the whole stack down the moment the backend vanished — so a mid-session
# CUDA illegal-memory-access (llama.cpp #23210, the crash in the report) ended
# the run and left Claude Code with no endpoint until a human re-ran start.sh.
#
# Now llama-server is supervised: an unexpected crash is relaunched in place
# (LiteLLM keeps its port bound the whole time, so the endpoint just blips
# during the reload instead of disappearing). Guard-initiated stops are handled
# by whether killing llama-server clears the condition (see is_terminal_guard_reason):
# memory-pressure and low-VRAM SELF-CLEAR (llama IS the scarce RSS/VRAM), so they
# restart by default; over-temp needs real cooldown so it stays terminal unless
# LLAMA_RESTART_ON_GUARD_STOP=1; a *-wedged passthrough is always terminal.
#
# Deterministic crash protection: a run only "counts as healthy" (resetting the
# failure streak) once it has served for LLAMA_HEALTHY_RESET_SEC. Faster crashes
# accumulate, and after LLAMA_MAX_RESTARTS consecutive fast failures the
# supervisor gives up rather than hammering a wedged GPU. Backoff grows
# geometrically between attempts, giving the CUDA driver time to settle.
# ---------------------------------------------------------------------------

read_guard_reason() {
    [ -f "$GUARD_REASON_FILE" ] && cat "$GUARD_REASON_FILE" 2>/dev/null || true
}

# Decide whether a guard stop is TERMINAL (give up) or recoverable (restart).
# The distinction is whether killing llama-server itself removes the condition:
#   - memory-pressure / gpu-low-vram: SELF-CLEARING. llama-server IS the ~24GB RSS
#     and the VRAM tenant, so killing it frees exactly what was scarce — a fresh
#     generation lands in a clean state. These now RESTART by default (with the
#     usual streak cap + backoff, so a genuine runaway still gives up after
#     LLAMA_MAX_RESTARTS instead of looping forever).
#   - gpu-overtemp: NOT self-clearing. The die is hot; it needs wall-clock cooldown,
#     not a 5s retry. Stays terminal unless the operator opts in.
#   - *-wedged: ALWAYS terminal. The GPU passthrough is stuck in an uninterruptible
#     ioctl; only `wsl.exe --shutdown` clears it, so retrying is futile and the
#     opt-in must NOT override it.
is_terminal_guard_reason() {
    case "$1" in
        *-wedged) return 0 ;;
        runtime-gpu-overtemp)
            [ "$LLAMA_RESTART_ON_GUARD_STOP" = "1" ] && return 1 || return 0 ;;
        runtime-system-memory-pressure|runtime-gpu-low-vram) return 1 ;;
        *) return 1 ;;
    esac
}

# Reap the current generation's watchdog before relaunching (or reading its
# reason file). It polls a specific PID, so once that PID is dead it exits on
# its own — but we wait on it so the reason it records is fully flushed to disk
# before the supervisor reads it, and so a stale watchdog never lingers.
reap_watchdog() {
    [ -n "$MONITOR_PID" ] || return 0
    kill -0 "$MONITOR_PID" 2>/dev/null && wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
}

# Sleep between restart attempts: base << (n-1), capped. Interruptible — a
# Ctrl+C during the wait still lands the INT trap and shuts things down.
restart_backoff() {
    local n="$1" delay="$LLAMA_RESTART_BACKOFF_BASE_SEC" i=1
    while [ "$i" -lt "$n" ]; do
        delay=$(( delay * 2 ))
        i=$(( i + 1 ))
    done
    [ "$delay" -gt "$LLAMA_RESTART_BACKOFF_MAX_SEC" ] && delay="$LLAMA_RESTART_BACKOFF_MAX_SEC"
    echo "==> Waiting ${delay}s for the GPU/driver to settle before relaunching..."
    log_guard "restart_backoff attempt=$n delay_s=$delay"
    sleep "$delay"
}

# LLAMA_RESTART_ENABLE=0 restores the original single-shot behavior: cap the
# budget at one generation so any failure falls straight through to the exit.
if [ "$LLAMA_RESTART_ENABLE" != "1" ]; then
    LLAMA_MAX_RESTARTS=1
fi

generation=0
consecutive_failures=0
supervise_exit_code=0

while true; do
    generation=$((generation + 1))
    echo "==> Starting llama-server on port $LLAMA_PORT (generation $generation)..."
    rm -f "$READY_FLAG_FILE" "$GUARD_REASON_FILE"

    if ! launch_llama_server "$generation"; then
        echo "==> llama-server failed to spawn — last lines of $LLAMA_LOG_FILE:" >&2
        tail -n 25 "$LLAMA_LOG_FILE" >&2 2>/dev/null || true
        GUARD_EXIT_REASON="llama-server-spawn-failed"
        log_guard "spawn_failed generation=$generation reason=$GUARD_EXIT_REASON"
        consecutive_failures=$((consecutive_failures + 1))
        if [ "$consecutive_failures" -ge "$LLAMA_MAX_RESTARTS" ]; then
            supervise_exit_code=1
            break
        fi
        restart_backoff "$consecutive_failures"
        continue
    fi

    log_guard "startup_ok generation=$generation pid=$LLAMA_PID ctx=$LLAMA_CTX batch=$LLAMA_BATCH_SIZE ubatch=$LLAMA_UBATCH_SIZE"
    [ "$generation" -eq 1 ] && [ "$RUNTIME_OOM_GUARD_ENABLE" = "1" ] \
        && echo "==> Runtime OOM guard enabled (min free ${RUNTIME_MIN_FREE_GB}GB, check every ${RUNTIME_CHECK_INTERVAL_SEC}s)."
    start_watchdog

    if ! wait_for_health; then
        # Died during load — never became healthy. This is the deterministic
        # startup-crash case; count it and let the streak logic cap retries.
        reap_watchdog
        reason=$(read_guard_reason)
        [ -z "$reason" ] && reason="llama-server-exited-before-health"
        GUARD_EXIT_REASON="$reason"
        log_guard "health_failed generation=$generation reason=$reason"
        if is_terminal_guard_reason "$reason"; then
            echo "==> Guard stop before ready ($reason) — not restarting. Set LLAMA_RESTART_ON_GUARD_STOP=1 to override."
            supervise_exit_code=1
            break
        fi
        consecutive_failures=$((consecutive_failures + 1))
        if [ "$consecutive_failures" -ge "$LLAMA_MAX_RESTARTS" ]; then
            echo "==> llama-server crashed during load $consecutive_failures times in a row — giving up." >&2
            echo "==> This usually means a deterministic startup failure (bad args, corrupt model, wedged GPU)." >&2
            echo "==> On WSL2 a wedged GPU passthrough often needs 'wsl.exe --shutdown' from PowerShell." >&2
            supervise_exit_code=1
            break
        fi
        restart_backoff "$consecutive_failures"
        continue
    fi

    log_guard "llama_ready generation=$generation health_ok=1"
    ready_ts=$(date +%s)

    # Start LiteLLM once, on the first healthy generation, and keep it up across
    # llama-server restarts so Claude Code's endpoint never disappears.
    if [ -z "$LITELLM_PID" ] || ! kill -0 "$LITELLM_PID" 2>/dev/null; then
        start_litellm
    fi

    # Watch both processes. Whichever dies first decides what happens next:
    #   - LiteLLM gone      → operator Ctrl+C or proxy crash: shut everything down.
    #   - llama-server gone → crash or guard stop: consult the reason and restart
    #                         or give up. The short sleep keeps signal traps
    #                         responsive (a foreground `wait` on the detached
    #                         server isn't possible — it's not our child).
    #   - watchdog gone but backend STILL alive → the guard fired, ran
    #     terminate_llama_pid to completion (SIGTERM→grace→SIGKILL) and STILL
    #     couldn't reap it: the backend is wedged in an uninterruptible GPU ioctl
    #     (WSL2 /dev/dxg). Polling `kill -0` on it would loop forever — that was
    #     the reported hang. Treat it as terminal and hand the operator the one
    #     thing that clears a wedged passthrough: `wsl.exe --shutdown`.
    backend_down=0
    backend_wedged=0
    while true; do
        if [ -n "$LITELLM_PID" ] && ! kill -0 "$LITELLM_PID" 2>/dev/null; then
            echo "==> LiteLLM proxy exited — shutting the stack down."
            GUARD_EXIT_REASON="litellm-exited"
            supervise_exit_code=0
            break 2
        fi
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            backend_down=1
            break
        fi
        if [ -n "$MONITOR_PID" ] && ! kill -0 "$MONITOR_PID" 2>/dev/null; then
            backend_wedged=1
            break
        fi
        sleep 2
    done

    if [ "$backend_wedged" -eq 1 ]; then
        reap_watchdog
        reason=$(read_guard_reason)
        [ -z "$reason" ] && reason="llama-server-wedged-unkillable"
        GUARD_EXIT_REASON="$reason"
        echo "==> llama-server did not die even after SIGKILL ($reason) — its GPU passthrough is wedged." >&2
        echo "==> This is the WSL2 /dev/dxg uninterruptible-sleep stall; a new process can't use the GPU" >&2
        echo "==> until it clears. Run 'wsl.exe --shutdown' from PowerShell (not a full reboot), then rerun." >&2
        log_guard "backend_wedged generation=$generation reason=$reason"
        supervise_exit_code=1
        break
    fi

    [ "$backend_down" -eq 1 ] || continue

    # llama-server is gone. Reap the watchdog first so its reason file is fully
    # written before we read it (it records the reason then exits).
    reap_watchdog
    reason=$(read_guard_reason)
    [ -z "$reason" ] && reason="llama-server-died-unexpectedly"
    GUARD_EXIT_REASON="$reason"
    rm -f "$READY_FLAG_FILE" "$GUARD_REASON_FILE"

    now=$(date +%s)
    served=$(( now - ready_ts ))
    log_guard "backend_down generation=$generation reason=$reason served_s=$served"

    if is_terminal_guard_reason "$reason"; then
        case "$reason" in
            *-wedged)
                echo "==> Runtime guard stopped llama-server ($reason) after ${served}s — not restarting." >&2
                echo "==> The GPU passthrough is wedged (unkillable ioctl); a fresh process can't use the GPU" >&2
                echo "==> until it clears. Run 'wsl.exe --shutdown' from PowerShell, then rerun." >&2 ;;
            runtime-gpu-overtemp)
                echo "==> Runtime guard stopped llama-server (GPU over-temp) after ${served}s — not restarting." >&2
                echo "==> The die needs real cooldown, not a quick retry. Let it cool (improve airflow/clocks)" >&2
                echo "==> then rerun, or set LLAMA_RESTART_ON_GUARD_STOP=1 to auto-retry anyway." >&2 ;;
            *)
                echo "==> Runtime guard stopped llama-server ($reason) after ${served}s — not restarting." >&2
                echo "==> Set LLAMA_RESTART_ON_GUARD_STOP=1 to auto-retry anyway." >&2 ;;
        esac
        supervise_exit_code=1
        break
    fi

    # A run that served long enough is considered stable — reset the streak so
    # occasional crashes hours apart don't eventually exhaust the retry budget.
    if [ "$served" -ge "$LLAMA_HEALTHY_RESET_SEC" ]; then
        consecutive_failures=0
    fi
    consecutive_failures=$((consecutive_failures + 1))

    echo "==> llama-server stopped after ${served}s (reason: $reason)."
    if [ "$consecutive_failures" -ge "$LLAMA_MAX_RESTARTS" ]; then
        echo "==> Reached $consecutive_failures consecutive failures without a stable run — giving up." >&2
        echo "==> On WSL2 a wedged GPU passthrough often needs 'wsl.exe --shutdown' from PowerShell." >&2
        supervise_exit_code=1
        break
    fi

    echo "==> Auto-restarting llama-server (attempt $((consecutive_failures + 1)) of $LLAMA_MAX_RESTARTS; LiteLLM stays up)."
    log_guard "auto_restart next_generation=$((generation + 1)) consecutive_failures=$consecutive_failures"
    # Make sure no half-dead process still holds GPU memory before relaunching.
    kill_stale_llama_server || true
    restart_backoff "$consecutive_failures"
done

# Fell out of the supervision loop — the EXIT trap (cleanup) frees GPU memory,
# stops LiteLLM and the watchdog, and propagates this code.
log_guard "supervisor_exit code=$supervise_exit_code reason=$GUARD_EXIT_REASON"
exit "$supervise_exit_code"
