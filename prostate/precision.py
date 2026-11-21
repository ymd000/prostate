"""Select the same autocast precision for training and inference."""

import torch


def resolve_precision(
    device: torch.device, amp: bool, requested: str = "auto"
) -> tuple[bool, torch.dtype]:
    if requested not in ("auto", "bfloat16", "float16"):
        raise ValueError("AMP dtype must be auto, bfloat16 or float16")
    if not amp or device.type != "cuda":
        return False, torch.float32
    supports_bfloat16 = torch.cuda.is_bf16_supported()
    if requested == "bfloat16" and not supports_bfloat16:
        raise ValueError("This GPU does not support BF16. Use --amp-dtype float16 or --no-amp.")
    use_bfloat16 = requested == "bfloat16" or (requested == "auto" and supports_bfloat16)
    return True, torch.bfloat16 if use_bfloat16 else torch.float16
