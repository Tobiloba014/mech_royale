"""Tracks weights.json versions, local scenario accuracy, and 48-match track scores."""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path("experiments.json")


def load_db() -> list[dict]:
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    return []


def save_db(records: list[dict]) -> None:
    DB_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def hash_files() -> str:
    h = hashlib.sha256()
    for fname in ("weights.json", "bot.py"):
        p = Path(fname)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


def log_submission(version: str, note: str, local_acc: float = 0.0):
    db = load_db()
    entry = {
        "version": version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hash": hash_files(),
        "note": note,
        "local_pass_pct": round(local_acc, 2),
        "track_score_48": None,
    }
    db.append(entry)
    save_db(db)
    print(f"Logged {version} (hash={entry['hash']}).")
    print(f"After 48 matches finish, run: python3 tracker.py score {version} <SCORE>")


def update_score(version: str, score: float):
    db = load_db()
    for row in db:
        if row["version"] == version:
            row["track_score_48"] = round(score, 4)
            save_db(db)
            print(f"Updated {version} -> 48-match mean score: {score:.4f} / 15.0")
            return
    print(f"Version '{version}' not found.")


def show_board():
    db = load_db()
    if not db:
        print("No experiments logged yet.")
        return
    print(f"{'VER':<10} {'HASH':<11} {'LOCAL%':<8} {'48-MATCH':<10} {'DELTA':<8} NOTE")
    print("-" * 78)
    prev_score = None
    for row in db:
        sc = row["track_score_48"]
        sc_str = f"{sc:.3f}" if sc is not None else "PENDING"
        if sc is not None and prev_score is not None:
            delta_str = f"{sc - prev_score:+.3f}"
        else:
            delta_str = "-"
        if sc is not None:
            prev_score = sc
        print(f"{row['version']:<10} {row['hash']:<11} {row['local_pass_pct']:<8.1f} {sc_str:<10} {delta_str:<8} {row['note']}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "log":
        acc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        log_submission(sys.argv[2], sys.argv[3], acc)
    elif len(sys.argv) >= 4 and sys.argv[1] == "score":
        update_score(sys.argv[2], float(sys.argv[3]))
    else:
        show_board()
