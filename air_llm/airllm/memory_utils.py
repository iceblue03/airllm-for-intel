import os
import sys
from pathlib import Path
from typing import Optional

import torch


def _warn(message: str):
    print(f"[airllm][memory] {message}")


def _parse_device_index(device: str) -> Optional[int]:
    if ":" not in device:
        return None
    try:
        return int(device.split(":", 1)[1])
    except Exception:
        return None


def get_available_memory_gb(device: str) -> float:
    try:
        if device.startswith("cuda"):
            index = _parse_device_index(device)
            free_mem = torch.cuda.mem_get_info(index)[0] if index is not None else torch.cuda.mem_get_info()[0]
            return float(free_mem) / (1024 ** 3)

        if device.startswith("xpu"):
            try:
                import intel_extension_for_pytorch as ipex  # noqa: F401
                device_idx = _parse_device_index(device) or 0
                props = torch.xpu.get_device_properties(device_idx)
                total = props.total_memory
                allocated = torch.xpu.memory_allocated(device)
                return float(total - allocated) / (1024 ** 3)
            except Exception:
                _warn("Cannot query XPU memory (IPEX not available or API mismatch). "
                      "Falling back to system RAM for memory estimation.")
                try:
                    import psutil
                    return float(psutil.virtual_memory().available) / (1024 ** 3)
                except Exception:
                    _warn("psutil is unavailable. Falling back to default memory estimate (4.0 GB).")
                    return 4.0

        if device.startswith("cpu"):
            try:
                import psutil
                return float(psutil.virtual_memory().available) / (1024 ** 3)
            except Exception:
                _warn("psutil is unavailable. Falling back to default memory estimate (4.0 GB).")
                return 4.0
    except Exception:
        pass

    try:
        import psutil
        return float(psutil.virtual_memory().available) / (1024 ** 3)
    except Exception:
        _warn("unable to detect available memory. Falling back to default memory estimate (4.0 GB).")
        return 4.0


def get_avg_layer_size_gb(checkpoint_path: str) -> float:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        _warn("checkpoint path is missing or does not exist. Using default average layer size (0.5 GB).")
        return 0.5

    base_path = Path(checkpoint_path)
    files = list(base_path.glob("*.safetensors"))
    if len(files) == 0:
        files = list(base_path.glob("*safetensors"))
    if len(files) == 0:
        _warn("no safetensors layer files found. Using default average layer size (0.5 GB).")
        return 0.5

    sizes = []
    for f in files:
        try:
            file_size = os.path.getsize(f)
            if file_size > 0:
                sizes.append(file_size)
        except Exception:
            pass

    if len(sizes) == 0:
        _warn("all detected layer files are zero-sized. Using default average layer size (0.5 GB).")
        return 0.5

    return (sum(sizes) / len(sizes)) / (1024 ** 3)


def suggest_num_layers(checkpoint_path: str, device: str, safety_margin_gb: float = 2.0) -> int:
    available = get_available_memory_gb(device)
    avg_size = get_avg_layer_size_gb(checkpoint_path)
    usable = available - safety_margin_gb
    if usable <= 0 or avg_size <= 0:
        return 1
    return max(1, int(usable / avg_size))


def calculate_min_required_memory_gb(avg_layer_size_gb: float, compression: Optional[str] = None) -> float:
    """Calculate minimum required memory (GB) to run the model.

    Parameters
    ----------
    avg_layer_size_gb : float
        Average on-disk size of one layer shard in GB.
    compression : str or None
        '4bit', '8bit', or None.

    Returns
    -------
    float
        Estimated minimum GPU/device memory required in GB.

    Notes
    -----
    Effective layer size is scaled by quantization:
    - '4bit'  → × 0.25
    - '8bit'  → × 0.5
    - None    → × 1.0
    Formula: min_required = effective_layer_size × 1.5 + 1.5
    (1.5× safety factor for double-buffering; +1.5 GB fixed overhead.)
    """
    if compression == "4bit":
        effective = avg_layer_size_gb * 0.25
    elif compression == "8bit":
        effective = avg_layer_size_gb * 0.5
    else:
        effective = avg_layer_size_gb
    return effective * 1.5 + 1.5


def check_memory_and_confirm(
    available_gb: float,
    min_required_gb: float,
    avg_layer_size_gb: float,
    compression: Optional[str] = None,
    device: Optional[str] = None,
) -> bool:
    """Warn the user when available memory is below the estimated minimum.

    If memory is sufficient, returns ``True`` immediately (no output).
    If memory is insufficient, prints a warning box and prompts Y/N.
    In a non-interactive environment (``sys.stdin.isatty() == False``), the
    warning is printed but execution continues automatically.

    Parameters
    ----------
    available_gb : float
        Currently available device memory in GB.
    min_required_gb : float
        Minimum required memory as returned by
        :func:`calculate_min_required_memory_gb`.
    avg_layer_size_gb : float
        Average layer shard size used to compute the estimate.
    compression : str or None
        Quantisation level shown in the warning box.
    device : str or None
        Device string shown in the warning box (e.g. ``"xpu:0"``).

    Returns
    -------
    bool
        Always ``True`` (or raises ``SystemExit(0)`` on user refusal).
    """
    if available_gb >= min_required_gb:
        return True

    device_str = device if device is not None else "unknown"

    print("┌──────────────────────────────────────────────────────┐")
    print("│  ⚠  메모리 경고                                      │")
    print("├──────────────────────────────────────────────────────┤")
    print(f"│  사용 가능한 메모리   :  {available_gb:.1f} GB  ({device_str})")
    print(f"│  모델 실행 권장 최소  :  {min_required_gb:.1f} GB")
    print(f"│  레이어당 평균 크기   :  {avg_layer_size_gb:.1f} GB")
    if compression is not None:
        print(f"│  양자화              :  {compression}  (메모리 절감 반영됨)")
    print("├──────────────────────────────────────────────────────┤")
    print("│  메모리 부족 시 추론 중 OOM이 발생할 수 있습니다.     │")
    print("└──────────────────────────────────────────────────────┘")

    if not sys.stdin.isatty():
        return True

    answer = input("그래도 계속 진행하시겠습니까? [Y/n]: ").strip().lower()
    if answer == "n":
        print("실행을 취소합니다.")
        raise SystemExit(0)
    return True


def confirm_num_layers(suggested: int, available_gb: float, avg_size_gb: float, safety_margin_gb: float = 2.0) -> int:
    suggested = max(1, int(suggested))

    if not sys.stdin.isatty():
        _warn(f"non-interactive environment detected; using suggested num_layers_in_memory={suggested}.")
        return suggested

    print("┌─────────────────────────────────────────────────────┐")
    print("│  AirLLM 메모리 자동 감지 결과                        │")
    print("├─────────────────────────────────────────────────────┤")
    print(f"│  사용 가능한 메모리   :  {available_gb:.1f} GB")
    print(f"│  시스템 예비 메모리   :  {safety_margin_gb:.1f} GB")
    print(f"│  레이어당 평균 크기   :  {avg_size_gb:.1f} GB")
    print(f"│  추천 동시 레이어 수  :  {suggested}")
    print("└─────────────────────────────────────────────────────┘")

    attempts = 0
    while attempts < 3:
        user_input = input(f"추천값 {suggested}으로 진행하시겠습니까? [Y/n/숫자 입력]: ").strip()
        if user_input == "" or user_input.lower() == "y":
            return suggested
        if user_input.lower() == "n":
            manual = input("직접 입력 (1 이상의 정수): ").strip()
            try:
                value = int(manual)
                if value >= 1:
                    return value
            except Exception:
                pass
            print("1 이상의 정수를 입력해주세요.")
            attempts += 1
            continue
        try:
            value = int(user_input)
            if value >= 1:
                return value
            print("1 이상의 정수를 입력해주세요.")
        except Exception:
            print("입력을 이해하지 못했습니다. Y/n 또는 숫자를 입력해주세요.")
        attempts += 1

    _warn(f"maximum retries exceeded; using suggested num_layers_in_memory={suggested}.")
    return suggested
