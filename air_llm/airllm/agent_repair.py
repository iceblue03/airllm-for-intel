#!/usr/bin/env python3
"""
agent_repair.py — install.sh 실행 중 오류 발생 시 AI가 터미널을 조작해서 자동으로 문제를 해결하는 에이전트.

사용법:
    python3 air_llm/airllm/agent_repair.py \
        --error "오류 메시지 전체" \
        --last-cmd "오류가 난 명령어" \
        --step "Step 번호 및 이름" \
        --repo-dir "/path/to/repo" \
        --venv-python "/path/to/.venv/bin/python"  # 선택, 없으면 python3 사용
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────
API_KEY_PATH = os.path.expanduser("~/.airllm/api_key.json")

BACKENDS = {
    "1": {"name": "OpenAI", "label": "ChatGPT, gpt-4o", "key": "openai", "package": "openai"},
    "2": {"name": "Anthropic", "label": "Claude, claude-3-5-sonnet", "key": "anthropic", "package": "anthropic"},
    "3": {"name": "Google", "label": "Gemini, gemini-1.5-pro", "key": "google", "package": "google-generativeai"},
}

MAX_SAME_ERROR = 5

# 위험 명령 패턴 목록
DANGEROUS_PATTERNS = [
    # 완전 차단 (Y/N 없이 무조건 거부)
    (r"sudo\s+rm\s+-[rRf]*\s+/(?!\S)", "BLOCK", "루트 디렉토리 삭제 시도 — 차단됨"),
    (r"rm\s+-[rRf]*\s+/(?!\S)", "BLOCK", "루트 디렉토리 삭제 시도 — 차단됨"),
    # Y/N 컨펌 필요
    (r"rm\s+-[rRf]", "CONFIRM", "재귀/강제 삭제"),
    (r"apt.*(remove|purge)", "CONFIRM", "패키지 제거"),
    (r"pip\s+uninstall", "CONFIRM", "pip 패키지 제거"),
    (r"chmod\s+[0-7]*7[0-7]*\s+/", "CONFIRM", "시스템 경로 권한 변경"),
    (r"chown\s+-R\s+root", "CONFIRM", "소유권 일괄 변경"),
    (r"systemctl\s+(stop|disable|mask)", "CONFIRM", "서비스 중단/비활성화"),
    (r"kill\s+-9", "CONFIRM", "프로세스 강제 종료"),
    (r">\s*/etc/", "CONFIRM", "/etc 파일 덮어쓰기"),
    (r"dd\s+", "CONFIRM", "dd 명령 (디스크 직접 쓰기)"),
    (r"mkfs", "CONFIRM", "파일시스템 포맷"),
]

SEP = "━" * 68


# ─────────────────────────────────────────────────────────────────────────────
# 안전 검사
# ─────────────────────────────────────────────────────────────────────────────
def check_safety(cmd: str):
    """
    Returns:
        ("BLOCK", reason)   — 무조건 차단
        ("CONFIRM", reason) — Y/N 컨펌 필요
        ("SAFE", "")        — 안전
    """
    for pattern, level, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return (level, reason)
    return ("SAFE", "")


# ─────────────────────────────────────────────────────────────────────────────
# 시스템 정보 수집
# ─────────────────────────────────────────────────────────────────────────────
def _run_silent(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return ""


def get_system_info(venv_python: str):
    os_info = _run_silent("lsb_release -ds 2>/dev/null || cat /etc/os-release | head -2")
    python_bin = venv_python if venv_python and os.path.isfile(venv_python) else sys.executable
    python_version = _run_silent(f"{python_bin} --version 2>&1")

    def disk_info(path):
        try:
            usage = shutil.disk_usage(path)
            return round(usage.free / 1e9, 1), round(usage.total / 1e9, 1)
        except Exception:
            return 0.0, 0.0

    disk_root_free, disk_root_total = disk_info("/")
    model_path = "/mnt/44026D66026D5DC4/Users/andre"
    disk_model_free, disk_model_total = disk_info(model_path) if os.path.exists(model_path) else (0.0, 0.0)

    try:
        import shutil
        mem_free_kb = int(_run_silent("awk '/MemAvailable/ {print $2}' /proc/meminfo") or "0")
        mem_total_kb = int(_run_silent("awk '/MemTotal/ {print $2}' /proc/meminfo") or "0")
        ram_free = round(mem_free_kb / 1e6, 1)
        ram_total = round(mem_total_kb / 1e6, 1)
    except Exception:
        ram_free, ram_total = 0.0, 0.0

    venv_dir = os.path.dirname(os.path.dirname(venv_python)) if venv_python else ""
    venv_exists = "예" if venv_dir and os.path.isdir(venv_dir) else "아니오"

    return {
        "os_info": os_info or "알 수 없음",
        "python_version": python_version or "알 수 없음",
        "disk_root_free": disk_root_free,
        "disk_root_total": disk_root_total,
        "disk_model_free": disk_model_free,
        "disk_model_total": disk_model_total,
        "ram_free": ram_free,
        "ram_total": ram_total,
        "venv_exists": venv_exists,
    }


# ─────────────────────────────────────────────────────────────────────────────
# API 키 관리
# ─────────────────────────────────────────────────────────────────────────────
def load_api_key():
    if os.path.isfile(API_KEY_PATH):
        try:
            with open(API_KEY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("backend", ""), data.get("key", "")
        except Exception:
            pass
    return "", ""


def save_api_key(backend: str, key: str):
    os.makedirs(os.path.dirname(API_KEY_PATH), exist_ok=True)
    with open(API_KEY_PATH, "w", encoding="utf-8") as f:
        json.dump({"backend": backend, "key": key}, f, ensure_ascii=False, indent=2)


def select_backend_and_key():
    print("AI 자동 수리 모드를 시작합니다.")
    print("사용할 AI 백엔드를 선택하세요:")
    print("  1) OpenAI  (ChatGPT, gpt-4o)")
    print("  2) Anthropic (Claude, claude-3-5-sonnet)")
    print("  3) Google  (Gemini, gemini-1.5-pro)")
    choice = input("선택 [1/2/3]: ").strip()
    if choice not in BACKENDS:
        print(f"[오류] 잘못된 선택: {choice}")
        sys.exit(1)
    backend_key = BACKENDS[choice]["key"]

    print(f"\nAPI 키를 입력하세요 (이후 {API_KEY_PATH} 에 저장됩니다):")
    api_key = input("> ").strip()
    if not api_key:
        print("[오류] API 키가 비어 있습니다.")
        sys.exit(1)

    save_api_key(backend_key, api_key)
    return backend_key, api_key


# ─────────────────────────────────────────────────────────────────────────────
# 패키지 자동 설치
# ─────────────────────────────────────────────────────────────────────────────
def ensure_package(package_name: str, venv_python: str):
    python_bin = venv_python if venv_python and os.path.isfile(venv_python) else sys.executable
    # Derive importable module name from the package name
    import_name = re.sub(r"[^a-zA-Z0-9_]", "_", package_name.split("[")[0])
    # Use importlib to check if the package is available (no shell injection risk)
    check = subprocess.run(
        [python_bin, "-c",
         f"import importlib.util; exit(0 if importlib.util.find_spec('{import_name}') is not None else 1)"],
        capture_output=True,
    )
    if check.returncode != 0:
        print(f"[패키지 설치] {package_name} 설치 중...")
        subprocess.run([python_bin, "-m", "pip", "install", package_name, "--break-system-packages"], check=True)


# ─────────────────────────────────────────────────────────────────────────────
# AI 백엔드별 요청
# ─────────────────────────────────────────────────────────────────────────────
def build_system_prompt(sysinfo: dict, step: str, last_cmd: str, error_msg: str, history: list) -> str:
    history_text = "\n".join(
        f"  {i + 1}. {cmd} → {result}" for i, (cmd, result) in enumerate(history)
    ) or "  (없음)"

    return (
        "당신은 Linux(Ubuntu/Linux Lite) 시스템에서 Intel XPU(Arc GPU) 환경을 설치하는 전문가입니다.\n"
        "사용자의 install.sh 실행 중 오류가 발생했습니다. 아래 정보를 바탕으로 문제를 진단하고\n"
        "실행 가능한 bash 명령어를 하나씩 제안해주세요.\n\n"
        "[시스템 상태]\n"
        f"- OS: {sysinfo['os_info']}\n"
        f"- Python: {sysinfo['python_version']}\n"
        f"- 디스크 (/): {sysinfo['disk_root_free']} GB 여유 / {sysinfo['disk_root_total']} GB 전체\n"
        f"- 디스크 (모델 저장 경로: /mnt/44026D66026D5DC4/Users/andre): "
        f"{sysinfo['disk_model_free']} GB 여유 / {sysinfo['disk_model_total']} GB 전체\n"
        f"- RAM: {sysinfo['ram_free']} GB 여유 / {sysinfo['ram_total']} GB 전체\n"
        f"- venv 존재: {sysinfo['venv_exists']}\n\n"
        "[오류 발생 Step]\n"
        f"{step}\n\n"
        "[마지막 실행 명령]\n"
        f"{last_cmd}\n\n"
        "[오류 메시지]\n"
        f"{error_msg}\n\n"
        "[지금까지 시도한 명령 이력]\n"
        f"{history_text}\n\n"
        "규칙:\n"
        '1. 반드시 JSON 형식으로만 응답하세요.\n'
        '2. 형식: {"cmd": "실행할 bash 명령어", "reason": "이 명령을 제안하는 이유"}\n'
        '3. 문제가 해결됐다고 판단되면: {"cmd": "SOLVED", "reason": "해결 이유"}\n'
        '4. 해결 불가능하다고 판단되면: {"cmd": "GIVE_UP", "reason": "포기 이유"}\n'
        "5. 한 번에 명령어 하나만 제안하세요."
    )


def ask_openai(api_key: str, prompt: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()
    return parse_ai_response(content)


def ask_anthropic(api_key: str, prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.content[0].text.strip()
    return parse_ai_response(content)


def ask_google(api_key: str, prompt: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    content = response.text.strip()
    return parse_ai_response(content)


def parse_ai_response(content: str) -> dict:
    # JSON 블록 추출 시도
    json_match = re.search(r"\{.*?\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    # 전체 파싱 시도
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"cmd": "GIVE_UP", "reason": f"AI 응답 파싱 실패: {content[:200]}"}


def ask_ai(backend: str, api_key: str, prompt: str) -> dict:
    print("[AI 분석 중...]")
    if backend == "openai":
        return ask_openai(api_key, prompt)
    elif backend == "anthropic":
        return ask_anthropic(api_key, prompt)
    elif backend == "google":
        return ask_google(api_key, prompt)
    else:
        print(f"[오류] 알 수 없는 백엔드: {backend}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 에이전트 루프
# ─────────────────────────────────────────────────────────────────────────────
def run_agent(error_msg: str, last_cmd: str, step: str, repo_dir: str, venv_python: str):
    print(f"\n{SEP}")
    print(f"  AI 자동 수리 모드 ({step})")
    print(SEP)

    # API 키 로드 또는 입력
    backend, api_key = load_api_key()
    if not backend or not api_key:
        backend, api_key = select_backend_and_key()

    # 백엔드 패키지 확인 및 설치
    backend_info = next((v for v in BACKENDS.values() if v["key"] == backend), None)
    if backend_info:
        ensure_package(backend_info["package"], venv_python)

    # 시스템 정보 수집
    sysinfo = get_system_info(venv_python)
    print(
        f"[시스템 상태] 디스크(/): {sysinfo['disk_root_free']}GB 여유  "
        f"RAM: {sysinfo['ram_free']}GB 여유"
    )

    history = []
    last_error_hash = ""
    retry_count = 0
    proposal_count = 0

    while True:
        # 시스템 프롬프트 생성
        prompt = build_system_prompt(sysinfo, step, last_cmd, error_msg, history)

        # AI에 제안 요청
        proposal_count += 1
        response = ask_ai(backend, api_key, prompt)
        cmd = response.get("cmd", "GIVE_UP")
        reason = response.get("reason", "")

        if cmd == "SOLVED":
            print(f"\n[AI 수리 완료] {reason}")
            sys.exit(0)

        if cmd == "GIVE_UP":
            print(f"\n[AI 수리 포기] {reason}")
            sys.exit(1)

        print(f"\n[AI 제안 #{proposal_count}] {cmd}")
        print(f"이유: {reason}")

        # 안전 검사
        safety, safety_reason = check_safety(cmd)
        if safety == "BLOCK":
            print(f"[차단] {safety_reason}")
            history.append((cmd, "BLOCKED"))
            continue

        if safety == "CONFIRM":
            answer = input(f"⚠️  경고: {safety_reason}\n정말 실행하시겠습니까? [y/N]: ")
            if answer.strip().lower() != "y":
                history.append((cmd, "USER_DENIED"))
                continue
        else:
            answer = input("실행하시겠습니까? [Y/n]: ")
            if answer.strip().lower() == "n":
                history.append((cmd, "USER_DENIED"))
                continue

        # 명령 실행
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        new_error = result.stderr if result.returncode != 0 else ""

        # 오류 해시 비교 → 카운트 관리
        new_hash = hashlib.md5(new_error.strip().encode()).hexdigest()
        if new_hash != last_error_hash:
            retry_count = 0
            last_error_hash = new_hash
        retry_count += 1

        if retry_count > MAX_SAME_ERROR:
            print(f"[AI 수리 실패] 동일 오류가 {MAX_SAME_ERROR}회 반복되었습니다.")
            print(f"마지막 오류:\n{new_error}")
            sys.exit(1)

        if result.returncode == 0:
            error_msg = ""
            print(f"[OK] 명령 성공: {cmd}")
            if result.stdout:
                print(result.stdout[:500])
        else:
            error_msg = new_error
            last_cmd = cmd
            print(f"[FAIL] 명령 실패: {cmd}")
            print(f"오류:\n{new_error[:500]}")

        history.append(
            (cmd, "OK" if result.returncode == 0 else f"FAIL: {new_error[:200]}")
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="install.sh 오류 발생 시 AI가 자동으로 문제를 해결하는 에이전트"
    )
    parser.add_argument("--error", required=True, help="오류 메시지 전체")
    parser.add_argument("--last-cmd", required=True, dest="last_cmd", help="오류가 난 명령어")
    parser.add_argument("--step", required=True, help="Step 번호 및 이름")
    parser.add_argument("--repo-dir", required=True, dest="repo_dir", help="리포지토리 경로")
    parser.add_argument("--venv-python", default="", dest="venv_python", help="venv python 경로 (선택)")
    args = parser.parse_args()

    run_agent(
        error_msg=args.error,
        last_cmd=args.last_cmd,
        step=args.step,
        repo_dir=args.repo_dir,
        venv_python=args.venv_python,
    )


if __name__ == "__main__":
    main()
