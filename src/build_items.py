"""Stage 0: build data/raw/items.json.

    python src/build_items.py --preset hard

Set B (items the model answers WRONG and stably) is the binding constraint. On
MMLU-professional + GSM8K a current frontier model yields ~3% Set B, which no
amount of scaling fixes - those benchmarks are simply too easy now. So the
default preset uses harder sources:

    MMLU-Pro       10 options, reasoning-heavy  -> replaces MMLU professional
    GPQA-diamond   graduate science, very hard  (198 items in total)
    MATH-500       competition maths            -> replaces GSM8K

`--preset easy` keeps the original MMLU/GSM8K sources, useful for weaker models
where Set B fills without difficulty.

Notes:
  * SAMPLING - splits are shuffled with a fixed seed before selection; benchmark
    files are often ordered, so taking the first N rows is a biased sample. Item
    ids are positions in the SHUFFLED order, so do NOT change --seed once
    screening has started, or the ids stop matching your runs.
  * PERTURBED VARIANTS - near-copies with reordered options, kept as separate
    items. Original vs perturbed accuracy is the contamination check.
  * Dataset ids change. Run --dry-run first to check every source loads.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string

from store import ROOT

OUT = os.path.join(ROOT, "data", "raw", "items.json")

# (domain, hf id, config, split, kind)
PRESETS = {
    "hard": [
        ("mmlupro_law", "TIGER-Lab/MMLU-Pro", None, "test", "mmlu_pro:law"),
        ("mmlupro_health", "TIGER-Lab/MMLU-Pro", None, "test", "mmlu_pro:health"),
        ("mmlupro_business", "TIGER-Lab/MMLU-Pro", None, "test", "mmlu_pro:business"),
        ("mmlupro_cs", "TIGER-Lab/MMLU-Pro", None, "test", "mmlu_pro:computer science"),
        ("gpqa_diamond", "Idavidrein/gpqa", "gpqa_diamond", "train", "gpqa"),
        ("math_500", "HuggingFaceH4/MATH-500", None, "test", "math"),
    ],
    "easy": [
        ("mmlu_law", "cais/mmlu", "professional_law", "test", "mcq_choices"),
        ("mmlu_medicine", "cais/mmlu", "professional_medicine", "test", "mcq_choices"),
        ("mmlu_accounting", "cais/mmlu", "professional_accounting", "test", "mcq_choices"),
        ("mmlu_security", "cais/mmlu", "computer_security", "test", "mcq_choices"),
        ("gsm8k_platinum", "madrylab/gsm8k-platinum", "main", "test", "gsm"),
    ],
}


def perturb_mcq(item: dict, rng: random.Random) -> dict:
    """Reorder options and relabel the answer: content identical, letter moved."""
    idx = list(range(len(item["options"])))
    rng.shuffle(idx)
    gold_old = string.ascii_uppercase.index(item["answer"])
    return {
        "id": item["id"] + "_p",
        "domain": item["domain"],
        "question": item["question"],
        "options": [item["options"][i] for i in idx],
        "answer": string.ascii_uppercase[idx.index(gold_old)],
        "perturbed_of": item["id"],
    }


def _extract_boxed(sol: str) -> str | None:
    m = re.search(r"\\boxed\{([^{}]+)\}", sol or "")
    return m.group(1).strip() if m else None


def build_rows(domain: str, kind: str, ds, limit: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for i, row in enumerate(ds):
        if len(rows) >= limit:
            break
        try:
            if kind.startswith("mmlu_pro"):
                cat = kind.split(":", 1)[1]
                if str(row.get("category", "")).lower() != cat.lower():
                    continue
                rows.append({"id": f"{domain}_{i:04d}", "domain": domain,
                             "question": row["question"], "options": list(row["options"]),
                             "answer": string.ascii_uppercase[int(row["answer_index"])]})

            elif kind == "gpqa":
                opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                        row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
                order = list(range(4))
                rng.shuffle(order)
                rows.append({"id": f"{domain}_{i:04d}", "domain": domain,
                             "question": row["Question"],
                             "options": [opts[j] for j in order],
                             "answer": string.ascii_uppercase[order.index(0)]})

            elif kind == "math":
                ans = _extract_boxed(row.get("solution")) or row.get("answer")
                if not ans:
                    continue
                rows.append({"id": f"{domain}_{i:04d}", "domain": domain,
                             "question": row["problem"], "answer": str(ans)})

            elif kind == "gsm":
                rows.append({"id": f"{domain}_{i:04d}", "domain": domain,
                             "question": row["question"],
                             "answer": row["answer"].split("####")[-1].strip().replace(",", "")})

            else:  # classic MMLU
                rows.append({"id": f"{domain}_{i:04d}", "domain": domain,
                             "question": row["question"], "options": list(row["choices"]),
                             "answer": string.ascii_uppercase[int(row["answer"])]})
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            print(f"  skipped {domain} row {i}: {exc}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["hard", "easy", "both"], default="hard")
    ap.add_argument("--per-source", type=int, default=250)
    ap.add_argument("--perturb", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--dry-run", action="store_true",
                    help="load 5 rows per source to check ids and field names; writes nothing")
    args = ap.parse_args()

    from datasets import load_dataset  # late import so --help works without it

    rng = random.Random(args.seed)
    sources = (PRESETS["hard"] + PRESETS["easy"]) if args.preset == "both" else PRESETS[args.preset]
    limit = 5 if args.dry_run else args.per_source
    items: list[dict] = []

    for domain, ds_id, cfg, split, kind in sources:
        try:
            ds = load_dataset(ds_id, cfg, split=split) if cfg else load_dataset(ds_id, split=split)
            ds = ds.shuffle(seed=args.seed)
        except Exception as exc:  # noqa: BLE001 - a dead id must not kill the build
            print(f"!! {domain}: could not load {ds_id} ({exc}). Fix the id in PRESETS.")
            continue
        rows = build_rows(domain, kind, ds, limit, rng)
        items += rows
        print(f"{domain}: {len(rows)} items")

    if args.dry_run:
        print("\ndry run - nothing written. Sample item:")
        print(json.dumps(items[0], indent=2) if items else "NO ITEMS - fix the dataset ids")
        return

    mcq = [it for it in items if it.get("options")]
    for it in rng.sample(mcq, int(len(mcq) * args.perturb)):
        items.append(perturb_mcq(it, rng))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(items, fh, indent=2)

    n_p = sum(1 for it in items if "perturbed_of" in it)
    print(f"\nwrote {OUT}: {len(items)} items ({n_p} perturbed)")
    print("Next: screen ONE strong model and read the printed 'B=' yield before")
    print("committing to the full build.")


if __name__ == "__main__":
    main()
