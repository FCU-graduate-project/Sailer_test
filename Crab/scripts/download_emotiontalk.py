"""
Download EmotionTalk (BAAI/Emotiontalk) from HuggingFace — the M3ED replacement.

Why this dataset (see BILINGUAL_FINETUNE_PLAN.md §5/§7):
  - Chinese multimodal (audio + text), 7 emotions = same scheme as M3ED
  - has Fear + Sad → maps to Unconfident (CH-SIMS could not provide this)
  - on HuggingFace → no Baidu Pan friction
  - license CC-BY-NC-SA-4.0 (NonCommercial) → academic/thesis use only

GATED dataset (gated: "auto"): you must
  1. log in to https://huggingface.co/datasets/BAAI/Emotiontalk and click
     "Agree and access" (auto-approved instantly)
  2. create a READ access token: https://huggingface.co/settings/tokens
  3. pass it here via --token, or `export HF_TOKEN=hf_xxx`, or `huggingface-cli login`

We only need Audio.tar (raw wav) + Text.tar (transcripts). Video.tar is skipped.

Usage:
  Crab/.venv/bin/python scripts/download_emotiontalk.py --token hf_xxx
  # then extract:
  Crab/.venv/bin/python scripts/download_emotiontalk.py --token hf_xxx --extract
"""
from pathlib import Path
import argparse
import os
import sys
import tarfile
import time

REPO_ID = "BAAI/Emotiontalk"
DEFAULT_OUT = Path("/home/brant/Project/SAILER_test/datasets/emotiontalk")
# Only the modalities Crab needs. Video.tar / Multimodal.tar skipped on purpose.
DEFAULT_FILES = ["Audio.tar", "Text.tar"]


def get_token(arg_token):
    tok = arg_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    # fall back to whatever `huggingface-cli login` stored
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None, help="HF read token (or set HF_TOKEN env / huggingface-cli login)")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="which tar files to fetch")
    ap.add_argument("--extract", action="store_true", help="extract downloaded tars in place")
    args = ap.parse_args()

    token = get_token(args.token)
    if not token:
        sys.exit(
            "ERROR: no HF token found.\n"
            "  1) accept terms at https://huggingface.co/datasets/BAAI/Emotiontalk\n"
            "  2) make a READ token at https://huggingface.co/settings/tokens\n"
            "  3) re-run with --token hf_xxx  (or export HF_TOKEN=hf_xxx)"
        )

    from huggingface_hub import hf_hub_download

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repo: {REPO_ID} (dataset)\nOut:  {args.out_dir}\nFiles: {args.files}\n")

    local_paths = []
    for fn in args.files:
        print(f"↓ downloading {fn} ...", flush=True)
        t0 = time.time()
        p = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=fn,
            token=token,
            local_dir=str(args.out_dir),
        )
        sz = Path(p).stat().st_size / 1e9
        print(f"  ✓ {fn}  {sz:.2f} GB  ({time.time()-t0:.0f}s)  → {p}")
        local_paths.append(Path(p))

    if args.extract:
        for p in local_paths:
            dest = args.out_dir / p.stem  # Audio.tar -> Audio/
            dest.mkdir(parents=True, exist_ok=True)
            print(f"⇲ extracting {p.name} → {dest} ...", flush=True)
            with tarfile.open(p) as tf:
                tf.extractall(dest)
            print(f"  ✓ extracted {p.name}")

    print("\n✅ done. Next: run scripts/build_emotiontalk_crab_csv.py --inspect "
          "to verify the internal layout, then build the CSV.")


if __name__ == "__main__":
    main()
