#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

RESULT_JSON="$(mktemp)"
REQUEST_JSON="$(mktemp)"
SERVER_SESSION_ID=""
INITIAL_POLL_DELAY_SECONDS="${INITIAL_POLL_DELAY_SECONDS:-15}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-3}"
CANCEL_REQUEST_TIMEOUT_SECONDS="${CANCEL_REQUEST_TIMEOUT_SECONDS:-5}"

trace_result() {
   # Persist a per-session diagnostic snapshot ONLY for genuine failures.
   # Rationale: this trace is debug-only (never read by training). Writing one
   # JSON per non-completed session on a shared filesystem accumulated hundreds
   # of thousands of tiny files across runs, making cleanup `rm -rf` take
   # minutes. "cancelled" sessions (partial-rollout aborts, etc.) are high-volume
   # and carry no diagnostic value, and every failure is already summarised in
   # agent_server_events.jsonl -- so we keep only STATUS=failed here.
   if [ -n "${AGENT_CLIENT_TRACE_DIR:-}" ] && [ "${STATUS}" = "failed" ]; then
      mkdir -p "${AGENT_CLIENT_TRACE_DIR}"
      cp "${RESULT_JSON}" "${AGENT_CLIENT_TRACE_DIR}/${RELAX_SESSION_ID}.${STATUS}.json"
   fi
}

cleanup() {
   rm -f "${RESULT_JSON}" "${REQUEST_JSON}"
}

cancel() {
   if [ -n "${SERVER_SESSION_ID}" ]; then
      curl -fsS --max-time "${CANCEL_REQUEST_TIMEOUT_SECONDS}" -X POST \
         "${AGENT_SERVER_URL}/sessions/${SERVER_SESSION_ID}/cancel" >/dev/null || true
   fi
   exit 143
}

trap cancel TERM INT
trap cleanup EXIT

jq -n \
   --arg session_id "${RELAX_SESSION_ID}" \
   --arg group_id "${RELAX_GROUP_ID}" \
   --arg mode "${RELAX_ROLLOUT_MODE}" \
   --arg base_url "${RELAX_BASE_URL}" \
   --arg api_key "${RELAX_SESSION_ID}" \
   '{session_id:$session_id,group_id:$group_id,mode:$mode,base_url:$base_url,api_key:$api_key}' \
   > "${REQUEST_JSON}"

curl -fsS -o "${RESULT_JSON}" -X POST "${AGENT_SERVER_URL}/sessions" \
   -H "Content-Type: application/json" --data-binary @"${REQUEST_JSON}"
SERVER_SESSION_ID="$(jq -r ".session_id" "${RESULT_JSON}")"

sleep "${INITIAL_POLL_DELAY_SECONDS}"

while true; do
   : > "${RESULT_JSON}"
   if curl -fsS -o "${RESULT_JSON}" "${AGENT_SERVER_URL}/sessions/${SERVER_SESSION_ID}"; then
      STATUS="$(jq -r ".status" "${RESULT_JSON}")"
      if [ "${STATUS}" = "completed" ] || [ "${STATUS}" = "failed" ] || [ "${STATUS}" = "cancelled" ]; then
         break
      fi
   fi
   sleep "${POLL_INTERVAL_SECONDS}"
done

if [ "${STATUS}" = "completed" ]; then
   jq '{metadata:{exit_status:.agent_exit_status,submission:.submission,n_calls:.n_calls},reward:.reward}' "${RESULT_JSON}" \
      > "${RELAX_OUTPUT_JSON}"
else
   trace_result
   jq -c . "${RESULT_JSON}" >&2
fi

exit "$(jq -r ".exit_code // 1" "${RESULT_JSON}")"
