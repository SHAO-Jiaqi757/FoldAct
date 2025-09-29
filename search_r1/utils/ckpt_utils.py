import os
import shutil
from typing import Optional


def find_latest_hf_checkpoint(ckpt_root: str) -> Optional[str]:
    """Return the latest HuggingFace checkpoint dir under an experiment folder.

    Expected layout:
      ckpt_root/
        latest_checkpointed_iteration.txt  (contains step number)
        global_step_{n}/actor/huggingface  (directory exists)

    If the tracker file is missing or invalid, falls back to the max numeric
    global_step_* folder that contains actor/huggingface.

    Returns absolute path to the HF dir, or None if not found.
    """
    if not ckpt_root:
        return None
    try:
        ckpt_root = os.path.abspath(ckpt_root)
        tracker = os.path.join(ckpt_root, "latest_checkpointed_iteration.txt")
        step: Optional[int] = None
        if os.path.isfile(tracker):
            try:
                content = open(tracker, "r", encoding="utf-8").read().strip()
                if content.isdigit():
                    step = int(content)
            except Exception:
                step = None

        def hf_dir_for(s: int) -> str:
            return os.path.join(ckpt_root, f"global_step_{s}", "actor", "huggingface")

        if step is not None:
            candidate = hf_dir_for(step)
            if os.path.isdir(candidate):
                return candidate

        # Fallback: scan all global_step_* dirs
        latest_path = None
        latest_step = -1
        try:
            for name in os.listdir(ckpt_root):
                if not name.startswith("global_step_"):
                    continue
                try:
                    s = int(name.split("global_step_")[-1])
                except ValueError:
                    continue
                p = hf_dir_for(s)
                if os.path.isdir(p) and s > latest_step:
                    latest_step = s
                    latest_path = p
        except FileNotFoundError:
            return None
        return latest_path
    except Exception:
        return None


def find_latest_actor_checkpoint(ckpt_root: str) -> Optional[str]:
    """Return the latest trainer actor checkpoint directory.

    Expected layout:
      ckpt_root/
        latest_checkpointed_iteration.txt  (contains step number)
        global_step_{n}/actor               (directory exists with model shards)

    If the tracker file is missing or invalid, falls back to the max numeric
    global_step_* folder that contains an 'actor' directory.
    """
    if not ckpt_root:
        return None
    try:
        ckpt_root = os.path.abspath(ckpt_root)
        tracker = os.path.join(ckpt_root, "latest_checkpointed_iteration.txt")
        step: Optional[int] = None
        if os.path.isfile(tracker):
            try:
                content = open(tracker, "r", encoding="utf-8").read().strip()
                if content.isdigit():
                    step = int(content)
            except Exception:
                step = None

        def actor_dir_for(s: int) -> str:
            return os.path.join(ckpt_root, f"global_step_{s}", "actor")

        if step is not None:
            candidate = actor_dir_for(step)
            if os.path.isdir(candidate):
                return candidate

        latest_path = None
        latest_step = -1
        try:
            for name in os.listdir(ckpt_root):
                if not name.startswith("global_step_"):
                    continue
                try:
                    s = int(name.split("global_step_")[-1])
                except ValueError:
                    continue
                p = actor_dir_for(s)
                if os.path.isdir(p) and s > latest_step:
                    latest_step = s
                    latest_path = p
        except FileNotFoundError:
            return None
        return latest_path
    except Exception:
        return None

def _parse_step_from_name(name: str) -> Optional[int]:
    if not name.startswith("global_step_"):
        return None
    try:
        return int(name.split("global_step_")[-1])
    except Exception:
        return None


def _list_all_steps(ckpt_root: str) -> list[int]:
    steps = []
    try:
        for name in os.listdir(ckpt_root):
            s = _parse_step_from_name(name)
            if s is not None:
                steps.append(s)
    except FileNotFoundError:
        pass
    return sorted(steps)


def delete_last_checkpoint(ckpt_root: str) -> Optional[int]:
    """Delete the latest checkpoint directory and update tracker.

    - Finds the max step under `ckpt_root` by reading the tracker file first, then
      falling back to scanning `global_step_*` directories.
    - Removes the entire `global_step_{last}` directory recursively.
    - Updates `latest_checkpointed_iteration.txt` to the next available lower step;
      if none remains, removes the tracker file.

    Returns the new latest step (int) if any remains, else None.
    """
    if not ckpt_root:
        return None
    ckpt_root = os.path.abspath(ckpt_root)
    tracker = os.path.join(ckpt_root, "latest_checkpointed_iteration.txt")

    # Determine last step
    last_step: Optional[int] = None
    if os.path.isfile(tracker):
        try:
            content = open(tracker, "r", encoding="utf-8").read().strip()
            if content.isdigit():
                last_step = int(content)
        except Exception:
            last_step = None

    if last_step is None:
        steps = _list_all_steps(ckpt_root)
        last_step = steps[-1] if steps else None
    if last_step is None:
        return None

    # Delete the last checkpoint directory
    last_dir = os.path.join(ckpt_root, f"global_step_{last_step}")
    if os.path.isdir(last_dir):
        shutil.rmtree(last_dir, ignore_errors=True)

    # Compute new latest step and update tracker accordingly
    remaining = [s for s in _list_all_steps(ckpt_root) if s < last_step]
    new_latest = remaining[-1] if remaining else None

    if new_latest is None:
        # No checkpoints remain; remove tracker if exists
        try:
            if os.path.isfile(tracker):
                os.remove(tracker)
        except Exception:
            pass
        return None
    else:
        # Write new tracker
        try:
            with open(tracker, "w", encoding="utf-8") as f:
                f.write(str(new_latest))
        except Exception:
            # If we fail to update tracker, still return computed step
            pass
        return new_latest
