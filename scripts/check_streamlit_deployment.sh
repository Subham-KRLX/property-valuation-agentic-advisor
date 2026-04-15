#!/usr/bin/env bash

set -euo pipefail

APP_URL="${APP_URL:-https://property-valuation-agentic-advisor-xvfy6pzq5caq72fmxlzrak.streamlit.app}"
USER_AGENT="${USER_AGENT:-Mozilla/5.0}"

health_response="$(curl -fsSL -A "$USER_AGENT" "${APP_URL}/healthz")"
if [[ "$health_response" != '{"status":"ok"}' ]]; then
  echo "::error::Health check failed for ${APP_URL}/healthz. Response: ${health_response}"
  exit 1
fi

echo "Health endpoint is healthy: ${health_response}"

root_headers="$(curl -sS -I -A "$USER_AGENT" "$APP_URL" || true)"

if grep -qi 'location: https://share.streamlit.io/-/auth/app' <<<"$root_headers"; then
  echo "::warning::Root URL is redirecting through Streamlit auth. The deployment is alive, but anonymous public access is not currently available."
else
  echo "Root URL did not redirect through Streamlit auth."
fi

