#!/bin/bash
#
# GoInfer 全链路 重新编译 + 启动脚本
#
# 与 start.sh 功能相同，但每次启动前会强制重新编译所有项目:
#   - Local-LLM C++ 引擎 (cmake + make)
#   - RAG Go 服务 (go build)
#   - LLM-Generation Go 服务 (go build)
#   - NLU Go 服务 (go build)
#
# 启动顺序 (按依赖关系):
#   1. Local-LLM C++ 引擎  :11434  (底层推理, 无依赖)
#   2. RAG 服务            :8081  (向量检索, 无依赖)
#   3. LLM-Generation 服务 :8082  (依赖 Local-LLM)
#   4. NLU 服务            :8080  (依赖 RAG + LLM-Generation)
#
# 用法:
#   ./rebuild-start.sh                    # 默认使用 14B 模型
#   ./rebuild-start.sh --model 7b         # 使用 7B 模型
#   ./rebuild-start.sh --model 1.5b       # 使用 1.5B 模型 (最快)
#   ./rebuild-start.sh --stop             # 停止所有服务
#   ./rebuild-start.sh --status           # 查看服务状态
#   ./rebuild-start.sh --test             # 启动后发送测试请求
#

set -euo pipefail

# ======================== 配置 ========================

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PID_DIR="$ROOT_DIR/.pids"

# 端口
PORT_LLM_ENGINE=11434
PORT_RAG=8081
PORT_LLM_GEN=8082
PORT_NLU=8080

# 模型映射
get_model_file() {
    case "$1" in
        14b)  echo "qwen2.5-coder-14b-instruct-q4_k_m.gguf" ;;
        7b)   echo "qwen2.5-coder-7b-instruct-q8_0.gguf" ;;
        1.5b) echo "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" ;;
        *)    echo "" ;;
    esac
}
VALID_MODELS="14b 7b 1.5b"
DEFAULT_MODEL="14b"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ======================== 工具函数 ========================

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}${CYAN}==> $*${NC}"; }

mkdir -p "$LOG_DIR" "$PID_DIR"

save_pid() {
    echo "$2" > "$PID_DIR/$1.pid"
}

read_pid() {
    local pid_file="$PID_DIR/$1.pid"
    if [[ -f "$pid_file" ]]; then
        cat "$pid_file"
    fi
}

is_running() {
    local pid
    pid=$(read_pid "$1")
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

wait_for_port() {
    local port=$1
    local name=$2
    local timeout=${3:-60}
    local elapsed=0

    while ! nc -z localhost "$port" 2>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [[ $elapsed -ge $timeout ]]; then
            log_error "$name 启动超时 (${timeout}s), 端口 $port 未就绪"
            log_error "查看日志: tail -50 $LOG_DIR/${name}.log"
            return 1
        fi
        # 每 5 秒打印进度
        if [[ $((elapsed % 5)) -eq 0 ]]; then
            printf "  等待 %s (%ds/%ds)...\r" "$name" "$elapsed" "$timeout"
        fi
    done
    echo ""  # 清除进度行
    return 0
}

# ======================== 停止服务 ========================

stop_all() {
    log_step "停止所有服务"

    local services=("nlu" "llm-generation" "rag" "localllm")
    for svc in "${services[@]}"; do
        if is_running "$svc"; then
            local pid
            pid=$(read_pid "$svc")
            log_info "停止 $svc (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            # 等待进程退出
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            # 如果还在运行, 强制杀
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$PID_DIR/$svc.pid"
            log_ok "$svc 已停止"
        else
            log_info "$svc 未在运行"
        fi
    done

    log_ok "所有服务已停止"
}

# ======================== 服务状态 ========================

show_status() {
    echo -e "\n${BOLD}GoInfer 服务状态${NC}"
    echo "────────────────────────────────────────────"
    printf "  %-20s %-8s %-10s %s\n" "服务" "端口" "状态" "PID"
    echo "────────────────────────────────────────────"

    local services=("localllm:$PORT_LLM_ENGINE" "rag:$PORT_RAG" "llm-generation:$PORT_LLM_GEN" "nlu:$PORT_NLU")
    for entry in "${services[@]}"; do
        local name="${entry%%:*}"
        local port="${entry##*:}"
        local pid status color

        pid=$(read_pid "$name")
        if is_running "$name"; then
            status="运行中"
            color="$GREEN"
        else
            status="已停止"
            color="$RED"
            pid="-"
        fi
        printf "  %-20s %-8s ${color}%-10s${NC} %s\n" "$name" ":$port" "$status" "$pid"
    done
    echo "────────────────────────────────────────────"
    echo ""
}

# ======================== 强制重新编译所有项目 ========================

rebuild_all() {
    local model_size=$1
    local build_start=$SECONDS

    log_step "重新编译所有项目"

    # 1. Local-LLM C++ 引擎
    log_info "编译 Local-LLM C++ 引擎..."
    mkdir -p "$ROOT_DIR/Local-LLM/build"
    (cd "$ROOT_DIR/Local-LLM/build" && cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null 2>&1 && make -j"$(sysctl -n hw.ncpu)" 2>&1) \
        || { log_error "Local-LLM C++ 编译失败"; exit 1; }
    log_ok "Local-LLM C++ 引擎编译完成"

    # 2. RAG Go 服务
    log_info "编译 RAG 服务..."
    (cd "$ROOT_DIR/RAG" && go build -o server ./cmd/server/) \
        || { log_error "RAG 编译失败"; exit 1; }
    log_ok "RAG 服务编译完成"

    # 3. LLM-Generation Go 服务
    log_info "编译 LLM-Generation 服务..."
    (cd "$ROOT_DIR/LLM-Generation" && go build -o server ./cmd/server/) \
        || { log_error "LLM-Generation 编译失败"; exit 1; }
    log_ok "LLM-Generation 服务编译完成"

    # 4. NLU Go 服务
    log_info "编译 NLU 服务..."
    (cd "$ROOT_DIR/NLU" && go build -o server ./cmd/server/) \
        || { log_error "NLU 编译失败"; exit 1; }
    log_ok "NLU 服务编译完成"

    local build_elapsed=$((SECONDS - build_start))
    log_ok "所有项目编译完成 (${build_elapsed}s)"

    # 检查模型文件
    local model_file
    model_file=$(get_model_file "$model_size")
    local model_path="$ROOT_DIR/Local-LLM/models/$model_file"
    if [[ ! -f "$model_path" ]]; then
        log_error "模型文件不存在: $model_path"
        log_info "可用模型:"
        ls -lh "$ROOT_DIR/Local-LLM/models/"*.gguf 2>/dev/null || echo "  (无)"
        exit 1
    fi

    log_ok "模型: $model_file ($(du -h "$model_path" | cut -f1))"
}

# ======================== 启动服务 (使用编译好的二进制) ========================

start_localllm() {
    local model_size=$1

    if is_running "localllm"; then
        log_warn "Local-LLM C++ 引擎已在运行 (PID: $(read_pid localllm))"
        return 0
    fi

    log_step "1/4 启动 Local-LLM C++ 引擎 (:$PORT_LLM_ENGINE)"

    local model_file
    model_file=$(get_model_file "$model_size")
    local model_path="$ROOT_DIR/Local-LLM/models/$model_file"

    "$ROOT_DIR/Local-LLM/build/localllm" \
        -m "$model_path" \
        -p "$PORT_LLM_ENGINE" \
        > "$LOG_DIR/localllm.log" 2>&1 &
    save_pid "localllm" $!

    log_info "PID: $!, 日志: $LOG_DIR/localllm.log"
    log_info "等待模型加载 (14B 约 5-15s, 1.5B 约 1-3s)..."

    # C++ 引擎加载模型需要时间
    local timeout=120
    if [[ "$model_size" == "1.5b" ]]; then
        timeout=30
    elif [[ "$model_size" == "7b" ]]; then
        timeout=60
    fi

    if wait_for_port "$PORT_LLM_ENGINE" "Local-LLM" "$timeout"; then
        # 再验证 health 端点
        sleep 1
        if curl -sf "http://localhost:$PORT_LLM_ENGINE/health" > /dev/null 2>&1; then
            log_ok "Local-LLM C++ 引擎就绪"
        else
            log_ok "Local-LLM C++ 引擎端口已开放 (health 端点待确认)"
        fi
    else
        log_error "Local-LLM 启动失败"
        tail -20 "$LOG_DIR/localllm.log"
        exit 1
    fi
}

start_rag() {
    if is_running "rag"; then
        log_warn "RAG 服务已在运行 (PID: $(read_pid rag))"
        return 0
    fi

    log_step "2/4 启动 RAG 服务 (:$PORT_RAG)"

    # 使用编译好的二进制而非 go run
    "$ROOT_DIR/RAG/server" \
        > "$LOG_DIR/rag.log" 2>&1 &
    save_pid "rag" $!

    log_info "PID: $!, 日志: $LOG_DIR/rag.log"

    if wait_for_port "$PORT_RAG" "RAG" 30; then
        log_ok "RAG 服务就绪"
    else
        log_warn "RAG 服务启动较慢, 继续启动其他服务 (NLU 会在 RAG 失败时跳过检索)"
    fi
}

start_llm_generation() {
    if is_running "llm-generation"; then
        log_warn "LLM-Generation 服务已在运行 (PID: $(read_pid llm-generation))"
        return 0
    fi

    log_step "3/4 启动 LLM-Generation 服务 (:$PORT_LLM_GEN)"

    # 使用编译好的二进制而非 go run
    (cd "$ROOT_DIR/LLM-Generation" && "$ROOT_DIR/LLM-Generation/server") \
        > "$LOG_DIR/llm-generation.log" 2>&1 &
    save_pid "llm-generation" $!

    log_info "PID: $!, 日志: $LOG_DIR/llm-generation.log"

    if wait_for_port "$PORT_LLM_GEN" "LLM-Generation" 30; then
        log_ok "LLM-Generation 服务就绪"
    else
        log_error "LLM-Generation 启动失败"
        tail -20 "$LOG_DIR/llm-generation.log"
        exit 1
    fi
}

start_nlu() {
    if is_running "nlu"; then
        log_warn "NLU 服务已在运行 (PID: $(read_pid nlu))"
        return 0
    fi

    log_step "4/4 启动 NLU 服务 (:$PORT_NLU)"

    # 使用编译好的二进制而非 go run
    (cd "$ROOT_DIR/NLU" && "$ROOT_DIR/NLU/server") \
        > "$LOG_DIR/nlu.log" 2>&1 &
    save_pid "nlu" $!

    log_info "PID: $!, 日志: $LOG_DIR/nlu.log"

    if wait_for_port "$PORT_NLU" "NLU" 30; then
        log_ok "NLU 服务就绪"
    else
        log_error "NLU 启动失败"
        tail -20 "$LOG_DIR/nlu.log"
        exit 1
    fi
}

# ======================== 测试 ========================

run_test() {
    log_step "发送测试请求"

    echo -e "${CYAN}请求:${NC} POST http://localhost:$PORT_NLU/api/v1/nlu/ask"
    echo -e "${CYAN}Body:${NC} {\"text\": \"写一个简单的go http服务器\", \"stream\": true}"
    echo ""

    # 流式请求
    curl -sN -X POST "http://localhost:$PORT_NLU/api/v1/nlu/ask" \
        -H 'Content-Type: application/json' \
        -d '{"text":"写一个简单的go http服务器","stream":true}' \
        --max-time 300 || {
        echo ""
        log_error "测试请求失败"
        log_info "检查日志:"
        echo "  tail -50 $LOG_DIR/nlu.log"
        echo "  tail -50 $LOG_DIR/llm-generation.log"
        echo "  tail -50 $LOG_DIR/localllm.log"
    }
    echo ""
}

# ======================== 主逻辑 ========================

main() {
    local model_size="$DEFAULT_MODEL"
    local do_test=false

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --stop)
                stop_all
                exit 0
                ;;
            --status)
                show_status
                exit 0
                ;;
            --test)
                do_test=true
                shift
                ;;
            --model)
                model_size="$2"
                if [[ -z "$(get_model_file "$model_size")" ]]; then
                    log_error "未知模型: $model_size (可选: $VALID_MODELS)"
                    exit 1
                fi
                shift 2
                ;;
            --14b|--14B)
                model_size="14b"
                shift
                ;;
            --7b|--7B)
                model_size="7b"
                shift
                ;;
            --1.5b|--1.5B)
                model_size="1.5b"
                shift
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo ""
                echo "与 start.sh 功能相同，但每次启动前会强制重新编译所有项目。"
                echo ""
                echo "选项:"
                echo "  --14b            使用 14B 模型 (默认)"
                echo "  --7b             使用 7B 模型"
                echo "  --1.5b           使用 1.5B 模型 (最快)"
                echo "  --model <size>   同上, 指定模型大小: 1.5b, 7b, 14b"
                echo "  --stop           停止所有服务"
                echo "  --status         查看服务状态"
                echo "  --test           启动后发送测试请求"
                echo "  --help           显示帮助"
                echo ""
                echo "示例:"
                echo "  $0 --14b              # 重新编译 + 启动 14B 模型全链路"
                echo "  $0 --1.5b --test      # 重新编译 + 启动 1.5B 并发送测试"
                echo "  $0 --stop             # 停止所有服务"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done

    echo -e "${BOLD}"
    echo "  ╔══════════════════════════════════════╗"
    echo "  ║    GoInfer 重新编译 + 全链路启动     ║"
    echo "  ╠══════════════════════════════════════╣"
    echo "  ║  Local-LLM C++  →  :$PORT_LLM_ENGINE            ║"
    echo "  ║  RAG            →  :$PORT_RAG             ║"
    echo "  ║  LLM-Generation →  :$PORT_LLM_GEN             ║"
    echo "  ║  NLU            →  :$PORT_NLU             ║"
    echo "  ╚══════════════════════════════════════╝"
    echo -e "${NC}"

    # 先停掉旧服务
    stop_all

    # 强制重新编译所有项目
    rebuild_all "$model_size"

    # 按依赖顺序启动
    local start_time=$SECONDS

    start_localllm "$model_size"
    start_rag
    start_llm_generation
    start_nlu

    local elapsed=$((SECONDS - start_time))

    # 显示最终状态
    echo ""
    echo -e "${BOLD}${GREEN}========================================${NC}"
    echo -e "${BOLD}${GREEN}  全链路启动完成 (${elapsed}s)${NC}"
    echo -e "${BOLD}${GREEN}========================================${NC}"

    show_status

    echo -e "${BOLD}使用方式:${NC}"
    echo ""
    echo "  # 非流式请求"
    echo "  curl -X POST http://localhost:$PORT_NLU/api/v1/nlu/ask \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"text\":\"写一个go服务器\"}'"
    echo ""
    echo "  # 流式请求 (推荐, 首 token 延迟更低)"
    echo "  curl -sN -X POST http://localhost:$PORT_NLU/api/v1/nlu/ask \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"text\":\"写一个go服务器\", \"stream\":true}'"
    echo ""
    echo -e "${BOLD}管理命令:${NC}"
    echo "  $0 --status     # 查看状态"
    echo "  $0 --stop       # 停止所有"
    echo "  $0 --test       # 发送测试"
    echo ""
    echo -e "${BOLD}日志:${NC}"
    echo "  tail -f $LOG_DIR/localllm.log"
    echo "  tail -f $LOG_DIR/nlu.log"
    echo ""

    if $do_test; then
        run_test
    fi
}

main "$@"
