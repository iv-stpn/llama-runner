#!/usr/bin/env bash
# Launch Claude Code pointed at the LiteLLM proxy (which fronts llama-server).
# Run after start.sh is up and the proxy is healthy.

export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="dummy"

export ANTHROPIC_MODEL="local"
export ANTHROPIC_SMALL_FAST_MODEL="local"

claude "$@"
