#!/usr/bin/env bash
# install.sh — AirLLM-for-Intel 원클릭 Intel XPU 환경 설치
#
# 사용법:
#   bash install.sh
#
# 전제조건:
#   - Ubuntu 22.04 기반 (Linux Lite 포함)
#   - Python 3.8+ 설치됨
#   - sudo 권한 있음
#   - 인터넷 연결
#
# 모델 다운로드는 포함되지 않습니다.
# 설치 완료 후 sudo reboot 권장.
# ─────────────────────────────────────────────────────────────────────────────
# AI 에이전트 설정
# ─────────────────────────────────────────────────────────────────────────────
REPO_DIR_EARLY="$(cd "$(dirname "$0")" && pwd)"
AGENT_PY="$REPO_DIR_EARLY/air_llm/airllm/agent_repair.py"

run_agent() {
    local error_msg="$1"
    local last_cmd="$2"
    local step_name="$3"
    local venv_py="${VENV_PYTHON:-}"

    if [ ! -f "$AGENT_PY" ]; then
        echo "[WARN] agent_repair.py를 찾을 수 없습니다: $AGENT_PY"
        return 1
    fi

    if [ -n "$venv_py" ] && [ -f "$venv_py" ]; then
        "$venv_py" "$AGENT_PY" \
            --error "$error_msg" \
            --last-cmd "$last_cmd" \
            --step "$step_name" \
            --repo-dir "$REPO_DIR_EARLY" \
            --venv-python "$venv_py"
    else
        python3 "$AGENT_PY" \
            --error "$error_msg" \
            --last-cmd "$last_cmd" \
            --step "$step_name" \
            --repo-dir "$REPO_DIR_EARLY"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: 색상/헬퍼 함수 정의
# ─────────────────────────────────────────────────────────────────────────────
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
SEP='━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
step() { echo -e "\n${SEP}\n  $*\n${SEP}"; }

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 시스템 패키지 설치
# ─────────────────────────────────────────────────────────────────────────────
step "Step 1: 시스템 패키지 설치"
if ! sudo apt-get update -y 2>/tmp/airllm_err.txt; then
    warn "apt-get update 실패. AI 수리 모드 시작..."
    run_agent "$(cat /tmp/airllm_err.txt)" "sudo apt-get update -y" "Step 1: 시스템 패키지 설치" || exit 1
    if ! sudo apt-get update -y 2>/tmp/airllm_err.txt; then
        err "AI 수리 후에도 apt-get update 실패"
    fi
fi
if ! sudo apt-get install -y \
  git curl wget gpg software-properties-common \
  build-essential cmake pkg-config \
  python3-venv python3-pip python3-dev \
  libssl-dev libffi-dev dkms 2>/tmp/airllm_err.txt; then
    warn "시스템 패키지 설치 실패. AI 수리 모드 시작..."
    run_agent "$(cat /tmp/airllm_err.txt)" \
        "sudo apt-get install -y git curl wget gpg software-properties-common build-essential cmake pkg-config python3-venv python3-pip python3-dev libssl-dev libffi-dev dkms" \
        "Step 1: 시스템 패키지 설치" || exit 1
    if ! sudo apt-get install -y \
      git curl wget gpg software-properties-common \
      build-essential cmake pkg-config \
      python3-venv python3-pip python3-dev \
      libssl-dev libffi-dev dkms 2>/tmp/airllm_err.txt; then
        err "AI 수리 후에도 시스템 패키지 설치 실패"
    fi
fi
ok "시스템 패키지 설치 완료"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Intel GPU 드라이버 저장소 등록
# ─────────────────────────────────────────────────────────────────────────────
step "Step 2: Intel GPU 드라이버 저장소 등록"
INTEL_REPO_LIST="/etc/apt/sources.list.d/intel.gpu.jammy.list"
if [ -f "$INTEL_REPO_LIST" ]; then
  info "Intel GPU 저장소가 이미 등록되어 있습니다. 건너뜁니다."
else
  if ! { wget -O- https://repositories.intel.com/graphics/intel-graphics.key \
    2>/tmp/airllm_err.txt \
    | sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics-archive-keyring.gpg 2>>/tmp/airllm_err.txt; }; then
    warn "Intel GPG 키 다운로드/등록 실패. AI 수리 모드 시작..."
    run_agent "$(cat /tmp/airllm_err.txt)" \
        "wget -O- https://repositories.intel.com/graphics/intel-graphics.key | sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics-archive-keyring.gpg" \
        "Step 2: Intel GPU 드라이버 저장소 등록" || exit 1
  fi
  if ! echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics-archive-keyring.gpg] https://repositories.intel.com/graphics/ubuntu jammy main" \
    | sudo tee "$INTEL_REPO_LIST" 2>/tmp/airllm_err.txt; then
    warn "Intel 저장소 목록 등록 실패. AI 수리 모드 시작..."
    run_agent "$(cat /tmp/airllm_err.txt)" \
        "echo '...' | sudo tee $INTEL_REPO_LIST" \
        "Step 2: Intel GPU 드라이버 저장소 등록" || exit 1
  fi
  if ! sudo apt-get update -y 2>/tmp/airllm_err.txt; then
    warn "Intel 저장소 추가 후 apt-get update 실패. AI 수리 모드 시작..."
    run_agent "$(cat /tmp/airllm_err.txt)" "sudo apt-get update -y" \
        "Step 2: Intel GPU 드라이버 저장소 등록" || exit 1
  fi
  ok "Intel GPU 저장소 등록 완료"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Intel GPU 드라이버 + 컴퓨트 런타임 설치
# ─────────────────────────────────────────────────────────────────────────────
step "Step 3: Intel GPU 드라이버 + 컴퓨트 런타임 설치"
info "Intel GPU 드라이버 패키지를 설치합니다 (일부 패키지가 없으면 경고 후 계속 진행)..."

sudo apt-get install -y intel-i915-dkms         || warn "intel-i915-dkms 패키지를 찾을 수 없습니다. 건너뜁니다."
sudo apt-get install -y intel-opencl-icd         || warn "intel-opencl-icd 패키지를 찾을 수 없습니다. 건너뜁니다."
sudo apt-get install -y intel-level-zero-gpu     || warn "intel-level-zero-gpu 패키지를 찾을 수 없습니다. 건너뜁니다."
sudo apt-get install -y intel-media-va-driver-non-free || warn "intel-media-va-driver-non-free 패키지를 찾을 수 없습니다. 건너뜁니다."
sudo apt-get install -y intel-igc-cm libigc1     || warn "intel-igc-cm / libigc1 패키지를 찾을 수 없습니다. 건너뜁니다."
sudo apt-get install -y intel-xpu-dkms           || true
sudo apt-get install -y intel-gsc                || true
ok "Intel GPU 드라이버 설치 단계 완료"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: render/video 그룹 권한 추가
# ─────────────────────────────────────────────────────────────────────────────
step "Step 4: render/video 그룹 권한 추가"
sudo usermod -aG render,video "$USER" 2>/dev/null || true
ok "그룹 추가 완료. 재부팅 또는 로그아웃 후 그룹이 적용됩니다."

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: /dev/dri/ 확인
# ─────────────────────────────────────────────────────────────────────────────
step "Step 5: /dev/dri/ 확인"
if [ -d /dev/dri ]; then
  info "/dev/dri/ 내용:"
  ls /dev/dri/
  if ! ls /dev/dri/ 2>/dev/null | grep -q "renderD128"; then
    warn "renderD128 장치가 없습니다. 드라이버 설치 후 재부팅이 필요할 수 있습니다."
  else
    ok "renderD128 장치 확인됨"
  fi
else
  warn "/dev/dri/ 디렉토리가 없습니다. 드라이버 설치 후 재부팅이 필요할 수 있습니다."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: venv 생성
# ─────────────────────────────────────────────────────────────────────────────
step "Step 6: Python 가상환경(venv) 생성"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ -d "$VENV_DIR" ]; then
  info "venv가 이미 존재합니다: $VENV_DIR — 건너뜁니다."
else
  if ! python3 -m venv "$VENV_DIR" 2>/tmp/airllm_err.txt; then
    warn "venv 생성 실패. AI 수리 모드 시작..."
    run_agent "$(cat /tmp/airllm_err.txt)" "python3 -m venv $VENV_DIR" \
        "Step 6: Python 가상환경(venv) 생성" || exit 1
    if ! python3 -m venv "$VENV_DIR" 2>/tmp/airllm_err.txt; then
        err "AI 수리 후에도 venv 생성 실패"
    fi
  fi
  ok "venv 생성 완료: $VENV_DIR"
fi

"$VENV_PIP" install --upgrade pip setuptools wheel
ok "pip/setuptools/wheel 업그레이드 완료"

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: torch + IPEX XPU 빌드 설치
# ─────────────────────────────────────────────────────────────────────────────
step "Step 7: torch + IPEX XPU 빌드 설치"
info "Intel 인덱스에서 torch/torchvision/torchaudio/intel-extension-for-pytorch 설치 중..."

install_torch() {
    "$VENV_PIP" install torch torchvision torchaudio intel-extension-for-pytorch \
        --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
}

if ! install_torch 2>/tmp/airllm_err.txt; then
    warn "torch 설치 실패. AI 수리 모드 시작..."
    if run_agent "$(cat /tmp/airllm_err.txt)" \
        "pip install torch torchvision torchaudio intel-extension-for-pytorch --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/" \
        "Step 7: torch + IPEX XPU 빌드 설치"; then
        if ! install_torch 2>/tmp/airllm_err.txt; then
            err "AI 수리 후에도 torch 설치 실패"
        fi
    else
        err "AI 수리 실패"
    fi
fi

info "XPU 빌드 여부 확인 중..."
if "$VENV_PYTHON" -c "import torch; v = torch.__version__; exit(0 if '+xpu' in v else 1)" 2>/tmp/airllm_err.txt; then
  ok "torch XPU 빌드 확인됨: $("$VENV_PYTHON" -c 'import torch; print(torch.__version__)')"
else
  warn "XPU 빌드가 아닌 torch가 설치되었습니다. AI 수리 모드 시작..."
  if run_agent "$(cat /tmp/airllm_err.txt)" \
      "pip install torch torchvision torchaudio intel-extension-for-pytorch --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/" \
      "Step 7: torch + IPEX XPU 빌드 설치"; then
      if ! "$VENV_PYTHON" -c "import torch; v = torch.__version__; exit(0 if '+xpu' in v else 1)"; then
          err "AI 수리 후에도 XPU 빌드 torch가 아닙니다. Intel 인덱스 접속을 확인하세요."
      fi
  else
      err "AI 수리 실패"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: bitsandbytes-intel 설치 + 심볼릭 링크 연결
# ─────────────────────────────────────────────────────────────────────────────
step "Step 8: bitsandbytes-intel 설치 + 심볼릭 링크 연결"
if ! "$VENV_PIP" install bitsandbytes-intel 2>/tmp/airllm_err.txt; then
    warn "bitsandbytes-intel 설치 실패. AI 수리 모드 시작..."
    if run_agent "$(cat /tmp/airllm_err.txt)" "pip install bitsandbytes-intel" \
        "Step 8: bitsandbytes-intel 설치 + 심볼릭 링크 연결"; then
        if ! "$VENV_PIP" install bitsandbytes-intel 2>/tmp/airllm_err.txt; then
            err "AI 수리 후에도 bitsandbytes-intel 설치 실패"
        fi
    else
        err "AI 수리 실패"
    fi
fi

BNB_XPU_SO=$(find "$VENV_DIR" -name "libbitsandbytes_xpu.so" 2>/dev/null | head -1)
IPEX_LIB_DIR=$(find "$VENV_DIR" -path "*/intel_extension_for_pytorch/lib" -type d 2>/dev/null | head -1)

if [ -n "$BNB_XPU_SO" ] && [ -n "$IPEX_LIB_DIR" ]; then
  ln -sf "$BNB_XPU_SO" "$IPEX_LIB_DIR/libintel-ext-pt-gpu-bitsandbytes.so"
  ok "bitsandbytes-intel 심볼릭 링크 연결 완료"
else
  info "libbitsandbytes_xpu.so 또는 IPEX lib 디렉토리를 찾지 못했습니다. 수동 확인 필요."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: 나머지 Python 패키지 설치
# ─────────────────────────────────────────────────────────────────────────────
step "Step 9: 나머지 Python 패키지 설치"
if ! "$VENV_PIP" install --upgrade \
  transformers \
  accelerate \
  safetensors \
  "huggingface_hub>=0.20.0" \
  psutil \
  sentencepiece \
  einops 2>/tmp/airllm_err.txt; then
    warn "Python 패키지 설치 실패. AI 수리 모드 시작..."
    if run_agent "$(cat /tmp/airllm_err.txt)" \
        "pip install --upgrade transformers accelerate safetensors huggingface_hub>=0.20.0 psutil sentencepiece einops" \
        "Step 9: 나머지 Python 패키지 설치"; then
        if ! "$VENV_PIP" install --upgrade \
          transformers \
          accelerate \
          safetensors \
          "huggingface_hub>=0.20.0" \
          psutil \
          sentencepiece \
          einops 2>/tmp/airllm_err.txt; then
            err "AI 수리 후에도 Python 패키지 설치 실패"
        fi
    else
        err "AI 수리 실패"
    fi
fi
ok "Python 패키지 설치 완료"

# ─────────────────────────────────────────────────────────────────────────────
# Step 10: XPU 인식 최종 확인
# ─────────────────────────────────────────────────────────────────────────────
step "Step 10: XPU 인식 최종 확인"
"$VENV_PYTHON" - << 'PYEOF'
import torch
import intel_extension_for_pytorch as ipex
print("torch   :", torch.__version__)
print("ipex    :", ipex.__version__)
print("xpu ok  :", torch.xpu.is_available())
if torch.xpu.is_available():
    print("xpu name:", torch.xpu.get_device_name(0))
PYEOF
XPU_CHECK=$("$VENV_PYTHON" -c "import torch; exit(0 if torch.xpu.is_available() else 1)" 2>/dev/null && echo "ok" || echo "fail")
if [ "$XPU_CHECK" = "ok" ]; then
  ok "XPU 인식 확인됨"
else
  warn "XPU가 인식되지 않습니다. 드라이버 설치 후 재부팅이 필요할 수 있습니다."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 11: 모델 저장 폴더 생성
# ─────────────────────────────────────────────────────────────────────────────
step "Step 11: 모델 저장 폴더 생성"
MODEL_DIR="/mnt/44026D66026D5DC4/Users/andre/airllm_models"
mkdir -p "$MODEL_DIR"
ok "모델 저장 폴더 생성: $MODEL_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Step 12: 완료 메시지
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "$SEP"
echo "  설치 완료!"
echo "$SEP"
echo ""
echo "  다음 단계:"
echo "  1. 재부팅 (드라이버/그룹 적용):"
echo "       sudo reboot"
echo ""
echo "  2. 재부팅 후, 리포 폴더로 이동 후 venv 활성화:"
echo "       cd $REPO_DIR"
echo "       . .venv/bin/activate"
echo ""
echo "  3. 진단 실행:"
echo "       python -m airllm.diagnostics run"
echo ""
echo "  4. 모델 다운로드 (예시: microsoft/Phi-3-mini-4k-instruct):"
echo "       python - << 'PY'"
echo "       import os"
echo "       from huggingface_hub import snapshot_download"
echo "       model_id = \"microsoft/Phi-3-mini-4k-instruct\""
echo "       save_to = \"/mnt/44026D66026D5DC4/Users/andre/airllm_models/\" + model_id.replace(\"/\", \"--\")"
echo "       snapshot_download(repo_id=model_id, local_dir=save_to, local_dir_use_symlinks=False)"
echo "       print(\"완료:\", save_to)"
echo "       PY"
echo ""
echo "  5. 채팅 실행:"
echo "       python -m airllm.chat"
echo "$SEP"
