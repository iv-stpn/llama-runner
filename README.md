# llama-runner

Run [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) locally
behind an Anthropic-compatible API, so Claude Code can use it as its model —
llama.cpp + LiteLLM on WSL2, hardened for a memory-constrained laptop.

```
Claude Code ──Anthropic Messages API──► LiteLLM proxy (:8000) ──OpenAI API──► llama-server (:8080)
```

Tuned for a modest rig: WSL2 on Windows, NVIDIA RTX 5070 Laptop (8 GB VRAM),
~29 GB RAM. MoE expert weights stay on CPU (`-cmoe`), attention runs on GPU.

## Usage

```bash
./start.sh              # builds llama.cpp on first run, downloads the model, starts both servers
./claude_code.sh        # launches Claude Code pointed at the local proxy
./test_api.sh           # smoke test: one question through the proxy
./start.sh --rebuild    # force-update and rebuild llama.cpp (pinned commit)
```

Stop with a single Ctrl+C and let it shut down gracefully — a 35B model can
take a while to release GPU memory, and interrupting CUDA teardown on WSL2
can wedge GPU passthrough until `wsl.exe --shutdown`.

## Files

| File | Purpose |
|------|---------|
| `start.sh` | Builds/starts llama-server + LiteLLM, with OOM/GPU/crash guards |
| `claude_code.sh` | Launches Claude Code with `ANTHROPIC_BASE_URL=http://localhost:8000` |
| `litellm_config.yaml` | LiteLLM proxy config (Anthropic → OpenAI translation) |
| `test_api.sh` | Smoke test via curl + python3 |

## Safety guards in start.sh

The script assumes the host is memory-constrained and actively defends
against WSL2 hard-crashes:

- **Startup memory precheck** — estimates model + KV-cache footprint and
  halves the context window (from `LLAMA_CTX_TARGET`, default 256k) until it
  fits in available RAM+swap; refuses to start if even the minimum won't fit.
- **WSL safety mode** — at large contexts (≥ `LLAMA_DISABLE_FA_CTX_THRESHOLD`,
  default 49152) disables flash-attention and speculative decoding and trims
  batch sizes.
- **Runtime watchdog** — stops the stack gracefully before the OS would
  OOM-kill it: system memory floor, GPU over-temperature, and a low-VRAM
  check that arms only after the server is healthy and baselines against
  the post-load steady state (on an 8 GB card, ~300 MB free is *normal*).
- **Crash telemetry** — `logs/llama_guard.log` records startup decisions,
  heartbeats and stop reasons; `logs/llama_server.log` captures llama-server
  output (rotated to `.prev` each run) so CUDA errors survive a lost terminal.

Every threshold is overridable via environment variables — see the top of
[start.sh](start.sh).

## Known issues

- **Speculative decoding (MTP) is disabled by default.** The `draft-mtp`
  path crashes with CUDA `illegal memory access` under multi-turn agentic
  load ([llama.cpp #23210](https://github.com/ggml-org/llama.cpp/issues/23210),
  still open). Once fixed upstream, re-enable with `LLAMA_SPEC_DRAFT_N_MAX=1`.
- **llama.cpp is pinned** to a verified commit (`LLAMA_CPP_REF` in start.sh).
  To try a newer upstream: `LLAMA_CPP_REF=master ./start.sh --rebuild`, and
  update the pin if it proves stable.

## Troubleshooting

- **Proxy port 8000 unreachable while llama-server runs** — usually a
  leftover LiteLLM from a previous run held the port, or a guard killed
  llama-server; check `logs/llama_guard.log` for the stop reason. start.sh
  now kills stale instances of both processes on startup.
- **Model load hangs at 0%** — WSL2 GPU passthrough may be wedged; run
  `wsl.exe --shutdown` from PowerShell and start again.
- **What killed the server?** — `grep runtime_stop logs/llama_guard.log`
  and `tail logs/llama_server.log`.
