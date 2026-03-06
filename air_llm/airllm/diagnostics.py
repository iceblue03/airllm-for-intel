"""AirLLM diagnostics helper — checks environment readiness and reports issues."""

import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time

STATUS_OK = "\u2705 OK"
STATUS_WARN = "\u26a0 \uacbd\uace0"
STATUS_ERROR = "\u274c \ubd88\uac00"
STATUS_UNKNOWN = "? \ud655\uc778\ubd88\uac00"

_DEFAULT_HISTORY_PATH = os.path.expanduser("~/.airllm/diagnostics_history.jsonl")


class DiagnosticsHistory:
    """진단 실행 기록을 JSON Lines 파일로 누적 저장·조회한다."""

    def __init__(self, path: str = _DEFAULT_HISTORY_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def record(self, event_type: str, data: dict) -> None:
        """이벤트를 한 줄의 JSON으로 기록한다.

        Parameters
        ----------
        event_type : str
            "diagnostics_run" | "package_update" | "package_downgrade" | "package_update_all"
        data : dict
            이벤트 상세 데이터 (버전, 성공여부, 메모리 등)
        """
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "event": event_type,
            **data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self) -> list:
        """저장된 모든 기록을 시간 오름차순으로 반환한다."""
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return sorted(entries, key=lambda e: e.get("timestamp", ""))

    def print_timeline(self) -> None:
        """라이브러리 업데이트/다운그레이드 시점과 진단 실행 결과를 시간 순서대로 출력한다."""
        entries = self.load()
        if not entries:
            print("기록된 진단 데이터가 없습니다.")
            return

        print()
        print("━" * 70)
        print("  AirLLM 진단 타임라인")
        print("━" * 70)

        for entry in entries:
            ts = entry.get("timestamp", "?")
            event = entry.get("event", "?")

            try:
                dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                ts_str = ts

            if event == "diagnostics_run":
                success = entry.get("success", None)
                icon = "✅" if success else ("❌" if success is False else "❓")
                elapsed = entry.get("elapsed_sec", None)
                elapsed_str = f" ({elapsed:.2f}s)" if elapsed is not None else ""
                print(f"\n[{ts_str}] {icon} 진단 실행{elapsed_str}")
                libs = entry.get("library_versions", {})
                if libs:
                    for lib, ver in libs.items():
                        print(f"    {lib:<30} {ver}")
                mem = entry.get("memory_gb", None)
                if mem:
                    print(f"    메모리 여유: {mem}")
                errors = entry.get("errors", [])
                if errors:
                    for err in errors:
                        print(f"    ❌ {err}")

            elif event in ("package_update", "package_downgrade", "package_update_all"):
                pkg = entry.get("package", "")
                from_ver = entry.get("from_version", "?")
                to_ver = entry.get("to_version", "?")
                success = entry.get("success", None)
                icon = "✅" if success else "❌"
                action_map = {
                    "package_update": "업데이트",
                    "package_downgrade": "다운그레이드",
                    "package_update_all": "전체 업데이트",
                }
                action = action_map.get(event, event)
                if event == "package_update_all":
                    pkg_str = entry.get("packages_summary", "")
                    print(f"\n[{ts_str}] {icon} 패키지 {action}: {pkg_str}")
                else:
                    print(f"\n[{ts_str}] {icon} {pkg} {action}: {from_ver} → {to_ver}")
                err = entry.get("error", None)
                if err:
                    print(f"    ❌ {err}")
            else:
                print(f"\n[{ts_str}] {event}: {entry}")

        print()
        print("━" * 70)


class PackageManager:
    """pip 기반 라이브러리 업데이트/다운그레이드 도우미.

    모든 작업은 DiagnosticsHistory에 자동 기록된다.
    """

    _ALIASES = {
        "bitsandbytes-intel": "bitsandbytes-intel",
        "bnb-intel": "bitsandbytes-intel",
        "bitsandbytes": "bitsandbytes",
    }

    def __init__(self, history: DiagnosticsHistory = None):
        self.history = history or DiagnosticsHistory()

    def _get_installed_version(self, package: str) -> str:
        """설치된 패키지 버전을 반환한다. 미설치 시 '미설치'를 반환한다."""
        try:
            import importlib.metadata
            return importlib.metadata.version(package)
        except Exception:
            return "미설치"

    def _run_pip(self, args: list) -> tuple:
        """pip 명령을 실행하고 (returncode, stdout, stderr)를 반환한다."""
        cmd = [sys.executable, "-m", "pip"] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr

    def update(self, package: str) -> bool:
        """특정 패키지를 최신 버전으로 업데이트한다.

        Parameters
        ----------
        package : str
            패키지 이름. 'bitsandbytes-intel' 등 별칭도 허용.

        Returns
        -------
        bool
            성공 여부.
        """
        package = self._ALIASES.get(package, package)
        from_ver = self._get_installed_version(package)
        print(f"📦 {package} 업데이트 중... (현재: {from_ver})")
        rc, out, err = self._run_pip(["install", "--upgrade", package])
        to_ver = self._get_installed_version(package)
        success = rc == 0
        self.history.record("package_update", {
            "package": package,
            "from_version": from_ver,
            "to_version": to_ver,
            "success": success,
            "error": err.strip() if not success else None,
        })
        if success:
            print(f"  ✅ {package}: {from_ver} → {to_ver}")
        else:
            print(f"  ❌ {package} 업데이트 실패:\n{err[:500]}")
        return success

    def downgrade(self, package: str, version: str) -> bool:
        """특정 패키지를 지정 버전으로 다운그레이드(또는 재설치)한다.

        Parameters
        ----------
        package : str
            패키지 이름.
        version : str
            목표 버전 (예: "0.39.0").

        Returns
        -------
        bool
            성공 여부.
        """
        package = self._ALIASES.get(package, package)
        from_ver = self._get_installed_version(package)
        target = f"{package}=={version}"
        print(f"📦 {package} 다운그레이드 중: {from_ver} → {version}")
        rc, out, err = self._run_pip(["install", target])
        to_ver = self._get_installed_version(package)
        success = rc == 0
        self.history.record("package_downgrade", {
            "package": package,
            "from_version": from_ver,
            "to_version": to_ver,
            "success": success,
            "error": err.strip() if not success else None,
        })
        if success:
            print(f"  ✅ {package}: {from_ver} → {to_ver}")
        else:
            print(f"  ❌ {package} 다운그레이드 실패:\n{err[:500]}")
        return success

    def update_all(self, packages: list = None) -> dict:
        """여러 패키지를 한꺼번에 최신 버전으로 업데이트한다.

        Parameters
        ----------
        packages : list, optional
            업데이트할 패키지 이름 목록. None이면 핵심 AirLLM 의존 패키지를 사용.

        Returns
        -------
        dict
            {package: True/False} 결과 맵.
        """
        if packages is None:
            packages = [
                "bitsandbytes-intel",
                "torch",
                "transformers",
                "accelerate",
                "intel_extension_for_pytorch",
                "psutil",
                "safetensors",
                "huggingface_hub",
            ]

        results = {}
        summary_parts = []
        print(f"📦 {len(packages)}개 패키지 전체 업데이트 시작...")
        for pkg in packages:
            ok = self.update(pkg)
            results[pkg] = ok
            summary_parts.append(f"{pkg}:{'OK' if ok else 'FAIL'}")

        self.history.record("package_update_all", {
            "packages": packages,
            "packages_summary": ", ".join(summary_parts),
            "success": all(results.values()),
        })
        return results


def _check_python():
    ver = platform.python_version()
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 8:
        return ("Python", STATUS_OK, ver, None)
    return ("Python", STATUS_WARN, ver, "Python 3.8 이상을 권장합니다.")


def _check_torch():
    try:
        import torch
        ver = torch.__version__
        return ("PyTorch", STATUS_OK, ver, None)
    except ImportError:
        return ("PyTorch", STATUS_ERROR, "미설치",
                "PyTorch가 설치되어 있지 않습니다. pip install torch 로 설치하세요.")


def _check_ipex():
    try:
        import intel_extension_for_pytorch as ipex
        ver = getattr(ipex, "__version__", "unknown")
        return ("Intel XPU 지원 (IPEX)", STATUS_OK, ver, None)
    except ImportError:
        return ("Intel XPU 지원 (IPEX)", STATUS_WARN, "미설치",
                "Intel GPU를 사용하려면 Intel Extension for PyTorch (IPEX)가\n"
                "  필요합니다. 설치 방법:\n"
                "    pip install intel_extension_for_pytorch\n"
                "  IPEX 없이도 CPU 모드로 실행은 가능합니다.")


def _check_xpu():
    try:
        import torch
        avail = torch.xpu.is_available() if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") else False
        if avail:
            return ("torch.xpu.is_available()", STATUS_OK, "True", None)
        return ("torch.xpu.is_available()", STATUS_ERROR, "False",
                "XPU 디바이스가 감지되지 않습니다. IPEX 설치 및 드라이버를 확인하세요.")
    except Exception as e:
        return ("torch.xpu.is_available()", STATUS_UNKNOWN, str(e), None)


def _check_cuda():
    try:
        import torch
        avail = torch.cuda.is_available()
        if avail:
            return ("torch.cuda.is_available()", STATUS_OK, "True", None)
        return ("torch.cuda.is_available()", STATUS_WARN, "False",
                "CUDA 디바이스가 감지되지 않습니다. NVIDIA GPU가 없거나 드라이버 문제일 수 있습니다.")
    except Exception as e:
        return ("torch.cuda.is_available()", STATUS_UNKNOWN, str(e), None)


def _check_memory(device=None):
    try:
        if device is not None and device.startswith("cuda"):
            import torch
            free = torch.cuda.mem_get_info()[0]
            val = f"{free / (1024**3):.1f} GB"
            return ("사용 가능한 메모리", STATUS_OK, val, None)
        if device is not None and device.startswith("xpu"):
            try:
                import torch
                import intel_extension_for_pytorch as ipex  # noqa: F401
                idx = int(device.split(":")[1]) if ":" in device else 0
                props = torch.xpu.get_device_properties(idx)
                total = props.total_memory
                allocated = torch.xpu.memory_allocated(device)
                free = total - allocated
                val = f"{free / (1024**3):.1f} GB"
                return ("사용 가능한 메모리", STATUS_OK, val, None)
            except Exception:
                pass
        # Fallback to system RAM
        import psutil
        avail = psutil.virtual_memory().available
        val = f"{avail / (1024**3):.1f} GB (시스템 RAM)"
        if avail / (1024**3) < 4.0:
            return ("사용 가능한 메모리", STATUS_WARN, val,
                    "사용 가능한 시스템 메모리가 4GB 미만입니다. 모델 로딩이 어려울 수 있습니다.")
        return ("사용 가능한 메모리", STATUS_OK, val, None)
    except Exception:
        return ("사용 가능한 메모리", STATUS_UNKNOWN, "확인 불가",
                "메모리 정보를 가져올 수 없습니다. psutil 설치를 확인하세요.")


def _check_disk(cache_dir=None):
    try:
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if not os.path.exists(cache_dir):
            cache_dir = os.path.expanduser("~")
        total, used, free = shutil.disk_usage(cache_dir)
        free_gb = free / (1024 ** 3)
        val = f"{free_gb:.1f} GB"
        if free_gb < 10.0:
            return ("디스크 여유 공간", STATUS_WARN, f"{val} (모델 로딩 불확실)",
                    f"디스크 여유 공간이 부족할 수 있습니다.\n"
                    f"  분할 저장 경로: {cache_dir}\n"
                    f"  권장 여유 공간: 모델 크기의 2배 이상")
        return ("디스크 여유 공간", STATUS_OK, val, None)
    except Exception:
        return ("디스크 여유 공간", STATUS_UNKNOWN, "확인 불가", None)


def _check_transformers():
    try:
        import transformers
        ver = transformers.__version__
        return ("transformers", STATUS_OK, ver, None)
    except ImportError:
        return ("transformers", STATUS_ERROR, "미설치",
                "transformers 라이브러리가 필요합니다. pip install transformers 로 설치하세요.")


def _check_psutil():
    try:
        import psutil
        ver = psutil.__version__
        return ("psutil", STATUS_OK, ver, None)
    except ImportError:
        return ("psutil", STATUS_WARN, "미설치",
                "psutil이 없으면 메모리 자동 감지가 불가능합니다.\n"
                "  pip install psutil 로 설치하세요.")


def _check_bitsandbytes():
    try:
        import bitsandbytes
        ver = getattr(bitsandbytes, "__version__", "unknown")
        return ("bitsandbytes", STATUS_OK, ver, None)
    except ImportError:
        return ("bitsandbytes", STATUS_WARN, "미설치 (압축 기능 불가)",
                "bitsandbytes가 없으면 4bit/8bit 압축을 사용할 수 없습니다.\n"
                "  압축 없이 실행하려면 compression=None으로 설정하세요.")


def _check_bitsandbytes_intel():
    """bitsandbytes-intel (Intel XPU용 4bit/8bit 압축) 설치 여부 확인."""
    try:
        import importlib.metadata
        ver = importlib.metadata.version("bitsandbytes-intel")
        return ("bitsandbytes-intel", STATUS_OK, ver, None)
    except Exception:
        return ("bitsandbytes-intel", STATUS_WARN, "미설치",
                "Intel XPU에서 4bit/8bit 압축을 사용하려면 bitsandbytes-intel이 필요합니다.\n"
                "  설치 방법: pip install bitsandbytes-intel\n"
                "  압축 없이 실행하려면 compression=None으로 설정하세요.")


def run_diagnostics(device=None, cache_dir=None, record_history=True):
    """Run all diagnostic checks and print a summary table + details.

    Parameters
    ----------
    device : str, optional
        Target device string (e.g. "cuda:0", "xpu:0", "cpu").
    cache_dir : str, optional
        HuggingFace cache directory for disk check.
    record_history : bool, optional
        Whether to record this run to DiagnosticsHistory. Default True.

    Returns
    -------
    dict
        Mapping of check name to (status, value, detail_or_none).
    """
    history = DiagnosticsHistory()
    t_start = time.time()

    checks = [
        _check_python,
        _check_torch,
        _check_ipex,
        _check_xpu,
        _check_cuda,
        lambda: _check_memory(device),
        lambda: _check_disk(cache_dir),
        _check_transformers,
        _check_psutil,
        _check_bitsandbytes,
        _check_bitsandbytes_intel,
    ]

    results = []
    for check_fn in checks:
        try:
            results.append(check_fn())
        except Exception as e:
            results.append(("?", STATUS_UNKNOWN, str(e), None))

    # Determine column widths
    col1_w = max(len(r[0]) for r in results)
    col2_w = max(len(r[1]) for r in results)
    col3_w = max(len(r[2]) for r in results)

    # Minimum widths
    col1_w = max(col1_w, 10)
    col2_w = max(col2_w, 6)
    col3_w = max(col3_w, 10)

    def _row(c1, c2, c3, sep="│"):
        return f"{sep} {c1:<{col1_w}} {sep} {c2:<{col2_w}} {sep} {c3:<{col3_w}} {sep}"

    line_w = col1_w + col2_w + col3_w + 10
    top = "┌" + "─" * (col1_w + 2) + "┬" + "─" * (col2_w + 2) + "┬" + "─" * (col3_w + 2) + "┐"
    mid = "├" + "─" * (col1_w + 2) + "┼" + "─" * (col2_w + 2) + "┼" + "─" * (col3_w + 2) + "┤"
    bot = "└" + "─" * (col1_w + 2) + "┴" + "─" * (col2_w + 2) + "┴" + "─" * (col3_w + 2) + "┘"

    print(top)
    print(_row("항목", "상태", "버전/값"))
    print(mid)
    for name, status, value, _ in results:
        print(_row(name, status, value))
    print(bot)

    # Detail section — only for warnings / errors
    details = [(name, status, value, detail) for name, status, value, detail in results if detail]
    if details:
        print()
        print("━" * line_w)
        for name, status, value, detail in details:
            label = "경고" if STATUS_WARN in status else "에러" if STATUS_ERROR in status else "정보"
            print(f"[{label}] {name} — {value}")
            for line in detail.split("\n"):
                print(f"  {line}")
            print()
        print("━" * line_w)

    elapsed = time.time() - t_start

    # 라이브러리 버전 스냅샷 수집
    lib_packages = ["torch", "transformers", "accelerate", "bitsandbytes", "bitsandbytes-intel",
                    "intel_extension_for_pytorch", "psutil", "safetensors"]
    lib_versions = {}
    for pkg in lib_packages:
        try:
            import importlib.metadata
            lib_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            lib_versions[pkg] = "미설치"

    # 에러 항목 수집
    error_items = [f"{name}: {value}" for name, status, value, _ in results
                   if STATUS_ERROR in status]

    # 메모리 정보 수집
    mem_result = next((r for r in results if r[0] == "사용 가능한 메모리"), None)
    mem_str = mem_result[2] if mem_result else None

    overall_success = not any(STATUS_ERROR in r[1] for r in results)

    if record_history:
        history.record("diagnostics_run", {
            "elapsed_sec": round(elapsed, 3),
            "device": device,
            "success": overall_success,
            "library_versions": lib_versions,
            "memory_gb": mem_str,
            "errors": error_items,
        })

    return {name: (status, value, detail) for name, status, value, detail in results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AirLLM 문제해결사")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("run", help="환경 진단 실행")
    sub.add_parser("timeline", help="진단 기록 타임라인 출력")

    p_update = sub.add_parser("update", help="패키지 업데이트")
    p_update.add_argument("package", help="패키지 이름 (예: bitsandbytes-intel)")

    p_down = sub.add_parser("downgrade", help="패키지 특정 버전으로 다운그레이드")
    p_down.add_argument("package", help="패키지 이름")
    p_down.add_argument("version", help="목표 버전 (예: 0.39.0)")

    p_all = sub.add_parser("update-all", help="핵심 패키지 전체 최신 업데이트")
    p_all.add_argument("--packages", nargs="*", help="업데이트할 패키지 목록 (미지정 시 기본 목록)")

    args = parser.parse_args()
    history = DiagnosticsHistory()
    pm = PackageManager(history=history)

    if args.cmd == "run" or args.cmd is None:
        run_diagnostics()
    elif args.cmd == "timeline":
        history.print_timeline()
    elif args.cmd == "update":
        pm.update(args.package)
    elif args.cmd == "downgrade":
        pm.downgrade(args.package, args.version)
    elif args.cmd == "update-all":
        pm.update_all(args.packages)
