#!/bin/bash
# Remote OpenAI-compatible LLM Judge 服务：使用外部托管的 OpenAI 兼容 endpoint 作为
# DeepEyes 的 LLM judge，替代 sglang_judge_service.sh 本地起 Qwen2.5-1.5B 的做法。
#
# 用法: source "$(dirname "${BASH_SOURCE[0]}")/openai_judge_service.sh"
#
# 与 sglang_judge_service.sh 的区别：本脚本不在本地启动任何服务，仅把 judge
# endpoint 通过环境变量（及 Ray runtime_env）透传给 reward worker。reward 代码
# （reward_deepeyes.py::_get_judge_client）原样消费 DEEPEYES_JUDGE_* 变量，无需改动。
#
# 可在调用前覆盖以下变量切换 endpoint / 模型 / api-key：
#   DEEPEYES_JUDGE_BASE_URL  (默认 http://29.160.40.138:8021/v1)
#   DEEPEYES_JUDGE_API_KEY   (默认 EMPTY)
#   DEEPEYES_JUDGE_MODELS    (默认 DeepSeek-V4-Flash-safety-t2)
# 依赖: TIMESTAMP 需在 source 前已定义（用于日志命名，可选）

# 设置默认值（允许外部覆盖）
DEEPEYES_JUDGE_BASE_URL="${DEEPEYES_JUDGE_BASE_URL:-http://29.160.40.138:8021/v1}"
DEEPEYES_JUDGE_API_KEY="${DEEPEYES_JUDGE_API_KEY:-EMPTY}"
DEEPEYES_JUDGE_MODELS="${DEEPEYES_JUDGE_MODELS:-DeepSeek-V4-Flash-safety-t2}"

# 检查必要依赖是否存在
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed."
    exit 1
fi

# 健康检查：远程 endpoint 不可达时 fail fast，避免训练跑到 reward 阶段才崩
echo "Checking remote LLM judge service at ${DEEPEYES_JUDGE_BASE_URL} ..."
http_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    -H "Authorization: Bearer ${DEEPEYES_JUDGE_API_KEY}" \
    "${DEEPEYES_JUDGE_BASE_URL}/models" 2>/dev/null || echo "000")
if [ "$http_status" != "200" ] && [ "$http_status" != "204" ]; then
    echo "Error: Remote LLM judge service not reachable at ${DEEPEYES_JUDGE_BASE_URL} (HTTP ${http_status})."
    echo "Set DEEPEYES_JUDGE_BASE_URL / DEEPEYES_JUDGE_API_KEY to a reachable OpenAI-compatible endpoint."
    exit 1
fi
echo "Remote LLM judge service is ready (models endpoint HTTP ${http_status})."
echo "  base_url: ${DEEPEYES_JUDGE_BASE_URL}"
echo "  models:   ${DEEPEYES_JUDGE_MODELS}"

# 导出到当前 shell（供非 Ray 路径 / 直接调用 reward 使用）
export DEEPEYES_JUDGE_API_KEY
export DEEPEYES_JUDGE_BASE_URL
export DEEPEYES_JUDGE_MODELS

# 注入 Ray runtime_env，使 reward worker（Ray actor）继承这些环境变量。
# 与 sglang_judge_service.sh 保持完全一致的注入方式。
if [ -n "${RUNTIME_ENV_JSON:-}" ]; then
    json_escape() {
        local value="${1:-}"
        value=${value//\\/\\\\}
        value=${value//\"/\\\"}
        value=${value//$'\n'/\\n}
        value=${value//$'\r'/\\r}
        value=${value//$'\t'/\\t}
        printf '%s' "$value"
    }

    runtime_env_prefix="${RUNTIME_ENV_JSON%$'\n}\n}'}"
    export RUNTIME_ENV_JSON="${runtime_env_prefix},
   \"DEEPEYES_JUDGE_API_KEY\": \"$(json_escape "${DEEPEYES_JUDGE_API_KEY}")\",
   \"DEEPEYES_JUDGE_BASE_URL\": \"$(json_escape "${DEEPEYES_JUDGE_BASE_URL}")\",
   \"DEEPEYES_JUDGE_MODELS\": \"$(json_escape "${DEEPEYES_JUDGE_MODELS}")\"
}
}"
fi

# 包装 ray：若调用方走 ray job submit 且未显式传 --runtime-env-json，则补上。
# 与 sglang_judge_service.sh 一致。
ray() {
    if [ "$1" = "job" ] && [ "$2" = "submit" ] && [ -n "${RUNTIME_ENV_JSON:-}" ]; then
        local arg=""
        for arg in "$@"; do
            if [ "$arg" = "--runtime-env-json" ] || [[ "$arg" == --runtime-env-json=* ]]; then
                command ray "$@"
                return
            fi
        done
        command ray job submit ${RAY_NO_WAIT:+--no-wait} --runtime-env-json="${RUNTIME_ENV_JSON}" "${@:3}"
        return
    fi
    command ray "$@"
}

echo "Remote LLM judge service is fully ready for use."
