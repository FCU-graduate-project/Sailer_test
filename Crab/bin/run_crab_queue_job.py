"""Single-job queue exec adapter. No campaign loop, GPU reassignment or W&B reset."""
import argparse
import os
from pathlib import Path
import sys

from crab_epoch_checkpoint import QueueContext, memory_guard, require


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER,
                        help="use -- followed by train_crab_queue.py arguments")
    parsed = parser.parse_args(argv)
    args = parsed.trainer_args
    if args[:1] == ["--"]:
        args = args[1:]
    QueueContext.from_env()
    memory_guard(start=True)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") and
            "," not in os.environ["CUDA_VISIBLE_DEVICES"], "One queue-assigned GPU required")
    # exec, not subprocess check=True: 75 is passed unchanged to the queue.
    # No WANDB_RESUME=never override, no deletion of WANDB_RUN_ID, no CUDA_VISIBLE_DEVICES=0.
    trainer = Path(__file__).with_name("train_crab_queue.py")
    os.execv(sys.executable, [sys.executable, "-B", str(trainer), *args])


if __name__ == "__main__":
    main()

