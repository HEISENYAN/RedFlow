from __future__ import annotations

import importlib
import sys

FLASH_ATTENTION_2_NAMES = {"flash_attention_2", "flash-attention-2"}


def configure_transformers_attn_runtime(attn_implementation: str | None):
    normalized = "" if attn_implementation is None else str(attn_implementation).strip().lower()
    if normalized in FLASH_ATTENTION_2_NAMES:
        try:
            importlib.import_module("flash_attn")
        except Exception as exc:  # pragma: no cover - depends on local binary env
            raise RuntimeError(
                "flash_attention_2 is configured as the active attention implementation, "
                "but the local flash_attn runtime cannot be imported. "
                "Please fix the flash-attn installation or binary compatibility before continuing."
            ) from exc
        return

    try:
        import transformers.utils as transformers_utils
        from transformers.utils import import_utils as transformers_import_utils
    except Exception:
        return

    def _disabled_flash_attn_check():
        return False

    transformers_import_utils.is_flash_attn_2_available = _disabled_flash_attn_check
    transformers_import_utils.is_flash_attn_3_available = _disabled_flash_attn_check
    if hasattr(transformers_utils, "is_flash_attn_2_available"):
        transformers_utils.is_flash_attn_2_available = _disabled_flash_attn_check
    if hasattr(transformers_utils, "is_flash_attn_3_available"):
        transformers_utils.is_flash_attn_3_available = _disabled_flash_attn_check
    sys.modules.pop("transformers.modeling_flash_attention_utils", None)
    sys.modules.pop("transformers.integrations.flash_attention", None)
