import glob
import os
import sys

def get_max_checkpoint(task_id):
    """
    Finds the maximum step number from saved checkpoints for a given task_id.
    Expected format: model_params/{task_id}_{step}_actor.pth

    Uses step NUMBER ordering (not mtime) to correctly identify the highest
    cumulative training step, even when phased training overwrites low-numbered
    checkpoints from earlier phases.
    """
    search_pattern = f'model_params/{task_id}_*_actor.pth'
    candidates = glob.glob(search_pattern)

    if not candidates:
        return 0

    max_step = 0
    for c in candidates:
        try:
            filename = os.path.basename(c)
            name_no_ext = filename.replace('_actor.pth', '')
            if name_no_ext.startswith(task_id):
                parts = name_no_ext.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    step = int(parts[1])
                    if step > max_step:
                        max_step = step
        except Exception:
            pass

    return max_step

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("0")
        sys.exit(0)

    task_id = sys.argv[1]
    print(get_max_checkpoint(task_id))
