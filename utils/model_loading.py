import os
import re
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _is_hf_model_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _maybe_actor_dir(path: str) -> Optional[str]:
    actor = os.path.join(path, "actor")
    if os.path.isdir(actor):
        # 优先检查 huggingface 子目录（VERL checkpoint 格式）
        hf_dir = os.path.join(actor, "huggingface")
        if os.path.isdir(hf_dir) and os.path.isfile(os.path.join(hf_dir, "config.json")):
            return hf_dir
        # 如果没有 huggingface 子目录，检查 actor 目录本身
        if os.path.isfile(os.path.join(actor, "config.json")):
            return actor
    return None


def _normalize_local_path(p: str) -> str:
    # Trim ends and collapse accidental whitespace-only segments between separators
    p = os.path.expanduser(str(p).strip())
    # Normalize segments to strip inner leading/trailing spaces around separators
    is_abs = p.startswith(os.sep)
    segments = [seg.strip() for seg in p.split(os.sep) if seg != ""]
    norm = os.path.join(*segments) if segments else ""
    if is_abs:
        norm = os.sep + norm
    return os.path.normpath(norm) if norm else p


def select_checkpoint_path(ckpt_root: str, step: Optional[str] = None) -> str:
    """
    Resolve a checkpoint directory under a training run root.

    - If `ckpt_root` itself is a valid HF model dir (has config.json), return it.
    - Else, look for subdirs matching global_step_* and pick by `step` or latest.
    """
    ckpt_root = _normalize_local_path(ckpt_root)

    if _is_hf_model_dir(ckpt_root):
        return ckpt_root
    actor = _maybe_actor_dir(ckpt_root)
    if actor:
        return actor

    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

    step_dirs = []
    pat = re.compile(r"^global_step_(\d+)$")
    for name in os.listdir(ckpt_root):
        m = pat.match(name)
        if m:
            full = os.path.join(ckpt_root, name)
            # Accept either HF dir at the step root, or an actor subdir with config
            if _is_hf_model_dir(full):
                step_dirs.append((int(m.group(1)), full))
            else:
                actor = _maybe_actor_dir(full)
                if actor:
                    step_dirs.append((int(m.group(1)), actor))
                # 也检查是否有直接的 huggingface 目录
                hf_dir = os.path.join(full, "huggingface")
                if os.path.isdir(hf_dir) and _is_hf_model_dir(hf_dir):
                    step_dirs.append((int(m.group(1)), hf_dir))

    if not step_dirs:
        raise FileNotFoundError(
            f"No HF checkpoints (config.json) found under {ckpt_root}. Expected subdirs like global_step_XX/"
        )

    step_dirs.sort(key=lambda x: x[0])
    if step is None or str(step).lower() == "latest":
        return step_dirs[-1][1]

    try:
        step_val = int(step)
    except ValueError:
        raise ValueError(f"Invalid step '{step}'. Use an integer or 'latest'.")

    for s, p in step_dirs:
        if s == step_val:
            return p

    available = ", ".join(str(s) for s, _ in step_dirs)
    raise FileNotFoundError(
        f"Requested step {step_val} not found under {ckpt_root}. Available steps: {available}"
    )


def load_causal_lm(
    model_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
    trust_remote_code: bool = True,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load tokenizer and causal LM from a directory or HF ID.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer, model
