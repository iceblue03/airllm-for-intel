from getpass import getpass

import torch

try:
    from .auto_model import AutoModel
    from .memory_utils import get_available_memory_gb, get_avg_layer_size_gb, suggest_num_layers, confirm_num_layers
except ImportError:
    from airllm.auto_model import AutoModel
    from airllm.memory_utils import get_available_memory_gb, get_avg_layer_size_gb, suggest_num_layers, confirm_num_layers


MAX_LENGTH = 512


def _default_device() -> str:
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        return "xpu:0"
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main():
    print("┌────────────────────────────────────────────────────────┐")
    print("│  AirLLM Chat Launcher                                  │")
    print("│  종료: Ctrl+C 또는 채팅 중 'exit' 입력                  │")
    print("└────────────────────────────────────────────────────────┘")

    try:
        model_id = ""
        while not model_id:
            print("모델 ID 또는 로컬 경로를 입력하세요.")
            print("  예시: meta-llama/Llama-3-8B")
            print("       microsoft/Phi-3-mini-4k-instruct")
            model_id = input("모델: ").strip()

        hf_token = None
        needs_token = input("HuggingFace 토큰이 필요한 모델인가요? [y/N]: ").strip().lower()
        if needs_token == "y":
            hf_token = getpass("HuggingFace 토큰 입력: ").strip() or None

        device = _default_device()
        available = get_available_memory_gb(device)
        avg_size = get_avg_layer_size_gb(model_id)
        suggested = suggest_num_layers(model_id, device)
        num_layers = confirm_num_layers(suggested, available, avg_size, safety_margin_gb=2.0)

        print("모델을 로딩 중입니다. 처음 실행 시 레이어 분할 저장으로 수 분이 걸릴 수 있습니다.")
        model = AutoModel.from_pretrained(
            model_id,
            device=device,
            hf_token=hf_token,
            num_layers_in_memory=num_layers,
        )

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("모델 로딩 완료. 채팅을 시작합니다.")
        print("※ 각 질문은 독립적으로 처리됩니다 (멀티턴 미지원).")
        print("종료하려면 'exit' 또는 Ctrl+C를 입력하세요.")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except KeyboardInterrupt:
                print("\n대화를 종료합니다.")
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "bye"}:
                print("대화를 종료합니다.")
                break

            try:
                print("생성 중...", end="\r", flush=True)
                input_tokens = model.tokenizer(
                    [user_input],
                    return_tensors="pt",
                    return_attention_mask=False,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=False,
                )
                input_ids = input_tokens["input_ids"].to(model.running_device)
                input_length = input_ids.shape[1]

                generation_output = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    use_cache=True,
                    return_dict_in_generate=True,
                )

                print(" " * 20, end="\r", flush=True)
                if not hasattr(generation_output, "sequences") or len(generation_output.sequences) == 0:
                    print("Assistant: (빈 응답)")
                    continue

                output_ids = generation_output.sequences[0][input_length:]
                response = model.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                print(f"Assistant: {response if response else '(빈 응답)'}")
            except KeyboardInterrupt:
                print("\n대화를 종료합니다.")
                break
            except Exception as e:
                print(f"오류가 발생했습니다: {e}")
                continue
    except KeyboardInterrupt:
        print("\n대화를 종료합니다.")


if __name__ == "__main__":
    main()
