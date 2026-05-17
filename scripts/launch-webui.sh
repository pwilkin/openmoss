#!/usr/bin/env bash
# Launch moss-tts-server with the bundled WebUI.
#
# Usage:
#   scripts/launch-webui.sh <model.gguf> [extra moss-tts-server args...]
#
# Environment overrides:
#   MOSS_SERVER  path to moss-tts-server binary (default: auto-detect)
#   MOSS_HOST    host to bind                   (default: 127.0.0.1)
#   MOSS_PORT    port to bind                   (default: 8080)
#   MOSS_WEBUI   path to webui directory        (default: auto-detect)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <model.gguf> [extra moss-tts-server args...]" >&2
    exit 2
fi

MODEL="$1"; shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Locate binary
if [[ -z "${MOSS_SERVER:-}" ]]; then
    for candidate in \
        "${PROJECT_DIR}/build/moss-tts-server" \
        "${PROJECT_DIR}/build/Release/moss-tts-server" \
        "${PROJECT_DIR}/moss-tts-server"; do
        if [[ -x "$candidate" ]]; then MOSS_SERVER="$candidate"; break; fi
    done
fi
if [[ -z "${MOSS_SERVER:-}" || ! -x "$MOSS_SERVER" ]]; then
    echo "error: moss-tts-server binary not found; set MOSS_SERVER or build the project first" >&2
    exit 1
fi

# Locate WebUI
if [[ -z "${MOSS_WEBUI:-}" ]]; then
    for candidate in \
        "${PROJECT_DIR}/webui" \
        "$(dirname "$MOSS_SERVER")/webui"; do
        if [[ -f "${candidate}/index.html" ]]; then MOSS_WEBUI="$candidate"; break; fi
    done
fi
if [[ -z "${MOSS_WEBUI:-}" ]]; then
    echo "error: webui directory not found; set MOSS_WEBUI" >&2
    exit 1
fi

HOST="${MOSS_HOST:-127.0.0.1}"
PORT="${MOSS_PORT:-8080}"

if [[ ! -f "$MODEL" ]]; then
    echo "warning: model file not found at '$MODEL' (passing through anyway)" >&2
fi

echo "────────────────────────────────────────────────"
echo " openmoss TTS — WebUI"
echo "   binary : $MOSS_SERVER"
echo "   model  : $MODEL"
echo "   webui  : $MOSS_WEBUI"
echo "   open   : http://${HOST}:${PORT}/"
echo "────────────────────────────────────────────────"

exec "$MOSS_SERVER" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --webui-dir "$MOSS_WEBUI" \
    "$@"
