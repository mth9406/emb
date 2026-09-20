"""Apply the confirmed IoU threshold band to mine_pairs.py's candidates.

Bounds were fixed by eyeballing samples per IoU band (see
notebooks/discussion_summary.md): below the lower bound the silhouettes are
often unrelated, above the upper bound the pair is usually the same photo
re-appearing under a different image_id.
"""

from __future__ import annotations

import argparse
import csv

from src.paths import DATA_ROOT

PAIRS_DIR = DATA_ROOT / "pairs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower", type=float, default=0.90)
    parser.add_argument("--upper", type=float, default=0.95)
    args = parser.parse_args()

    with (PAIRS_DIR / "candidates.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    kept = [r for r in rows if args.lower <= float(r["iou_score"]) < args.upper]

    with (PAIRS_DIR / "mined_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anchor_image_id", "positive_image_id", "iou_score"])
        writer.writeheader()
        writer.writerows(kept)

    anchors_total = len({r["anchor_image_id"] for r in rows})
    anchors_kept = len({r["anchor_image_id"] for r in kept})
    print(f"candidates {len(rows)} -> mined_pairs {len(kept)} (IoU in [{args.lower}, {args.upper}))")
    print(f"anchors with >=1 positive: {anchors_kept} / {anchors_total}")
    print(f"wrote {PAIRS_DIR / 'mined_pairs.csv'}")


if __name__ == "__main__":
    main()
