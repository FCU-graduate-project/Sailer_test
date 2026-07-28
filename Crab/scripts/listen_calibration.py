"""
Quick listen-check for the 5 calibration clips before deploying to raters.

Prints each clip's expected score + rationale, then plays it via the system
default audio player (or `ffplay` if available). After listening to all 5,
manually mark `verified_by_listening` = TRUE in calibration_set.csv if the
clip actually sounds the way the rationale expects.

If a clip doesn't sound extreme enough → swap with another (same emotion,
adjacent alpha, or different speaker) and re-listen.
"""
import csv
import subprocess
import sys
from pathlib import Path

CSV_PATH = Path("/home/brant/Project/SAILER_test/Crab/data/calibration_set.csv")


def play(wav_path):
    """Try ffplay → fall back to aplay → fall back to xdg-open."""
    candidates = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path],
        ["aplay", "-q", wav_path],
        ["paplay", wav_path],
        ["xdg-open", wav_path],
    ]
    for cmd in candidates:
        try:
            subprocess.run(cmd, check=True, timeout=30)
            return cmd[0]
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    print(f"⚠ no audio player worked. Try: ffplay {wav_path}")
    return None


def main():
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} calibration clips\n")

    for i, r in enumerate(rows, 1):
        print("=" * 70)
        print(f"[{i}/{len(rows)}] cal_id={r['cal_id']}")
        print(f"     expected score: {r['expected_score']}  ({r['reason']})")
        print(f"     pick rationale: {r['rationale_for_pick']}")
        print(f"     wav: {r['wav_path']}")
        print()
        input("     Press Enter to play (Ctrl+C to abort)…")
        used = play(r['wav_path'])
        if used:
            print(f"     [played via {used}]")
        print()
        answer = input(
            "     Does it actually sound the expected score? (y/n/replay): "
        ).strip().lower()
        while answer == "replay":
            play(r['wav_path'])
            answer = input(
                "     Does it actually sound the expected score? (y/n/replay): "
            ).strip().lower()
        r["verified_by_listening"] = "TRUE" if answer == "y" else "FALSE"
        if answer != "y":
            r["_listener_note"] = input("     Note (optional, what's wrong?): ").strip()

    # write back
    fieldnames = list(rows[0].keys())
    if "_listener_note" not in fieldnames:
        fieldnames.append("_listener_note")
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    verified = sum(1 for r in rows if r.get("verified_by_listening") == "TRUE")
    print(f"\n✅ {verified}/{len(rows)} calibration clips verified by listening")
    if verified < 5:
        print("⚠ Re-pick the unverified clips (try different speaker / alpha) before deploying to raters.")


if __name__ == "__main__":
    main()
