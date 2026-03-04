"""AirLLM diagnostics helper — checks environment readiness and reports issues."""

import os
import platform
import shutil
import sys

STATUS_OK = "\u2705 OK"
STATUS_WARN = "\u26a0 \uacbd\uace0"
STATUS_ERROR = "\u274c \ubd88\uac00"
STATUS_UNKNOWN = "? \ud655\uc778\ubd88\uac00"


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


def run_diagnostics(device=None, cache_dir=None):
    """Run all diagnostic checks and print a summary table + details.

    Parameters
    ----------
    device : str, optional
        Target device string (e.g. "cuda:0", "xpu:0", "cpu").
    cache_dir : str, optional
        HuggingFace cache directory for disk check.

    Returns
    -------
    dict
        Mapping of check name to (status, value, detail_or_none).
    """
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

    return {name: (status, value, detail) for name, status, value, detail in results}
