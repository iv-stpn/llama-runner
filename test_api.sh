#!/usr/bin/env bash
# Quick smoke test: ask the Anthropic-compatible proxy a basic question.
# Run after start.sh is up and the proxy is healthy.

set -euo pipefail

PROXY_URL="http://localhost:8000"
QUESTION="${1:-What is the capital of France? Answer in one short sentence.}"

curl -sf "$PROXY_URL/v1/messages" \
    -H "x-api-key: dummy" \
    -H "anthropic-version: 2023-06-01" \
    -H "content-type: application/json" \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"local","max_tokens":1024,"messages":[{"role":"user","content":sys.argv[1]}]}))' "$QUESTION")" \
    | python3 -c '
import json, sys
# Qwen3.6 is a thinking model: the content array can hold a thinking block
# before (or instead of) the text block, so pick out the text blocks only.
data = json.load(sys.stdin)
text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
print(text.strip() if text.strip() else "[no text block in response]\n" + json.dumps(data, indent=2))
'
