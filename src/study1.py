"""Study 1: sycophancy under user pushback on benchmark questions.

Usage:
    python src/study1.py --models gpt6_luna claude_sonnet5 --system baseline
    python src/study1.py --models gpt6_luna --system evidence_gate   # mitigation replay

The mitigation replay reuses the SAME items, pressure types, tiers and variant
seeds; only the system prompt differs. That is what makes the comparison clean.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import yaml

from collections import Counter

from conversation import run_conversation, schedule
from judge import Judge
from providers import Registry
from stimuli import load_bank
from store import ROOT, append, conversation_id, done_ids

SETS_DIR = os.path.join(ROOT, "data", "sets")
OBJ_DIR = os.path.join(ROOT, "data", "stimuli", "objections")
SRC_DIR = os.path.join(ROOT, "data", "stimuli", "sources")


def load_items(model_key: str) -> list[dict]:
    """Join the model's sets to the item catalogue.

    The screening records store only ids and answers - NOT the question text or
    the options. Reading `question`/`options` from them yields empty prompts, so
    the model is asked a blank question and every reply codes as drift. Always
    join to data/raw/items.json.
    """
    with open(os.path.join(ROOT, "data", "raw", "items.json")) as fh:
        catalogue = {i["id"]: i for i in json.load(fh)}
    with open(os.path.join(SETS_DIR, f"{model_key}.json")) as fh:
        d = json.load(fh)

    items, missing = [], []
    for which in ("set_a", "set_b"):
        for r in d[which]:
            src = catalogue.get(r["item_id"])
            if src is None:
                missing.append(r["item_id"])
                continue
            items.append({
                "item_id": r["item_id"], "domain": r["domain"], "gold": r["gold"],
                "question": src["question"], "options": src.get("options"),
                "pushed_answer": r["pushed_answer"], "set": which.upper()[-1],
                "confidence": r.get("confidence"),
            })
    if missing:
        raise SystemExit(
            f"{len(missing)} items are not in data/raw/items.json (e.g. {missing[:3]}).\n"
            "The item pool has changed since screening - do NOT run: ids must match.")
    return items


def _aux(model_key: str, item: dict, ptype: str) -> tuple[str | None, str | None]:
    """Per-item objection text and citation source, produced in stage 3."""
    if ptype == "objection":
        path = os.path.join(OBJ_DIR, f"{model_key}.json")
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh).get(item["item_id"], {}).get(item["set"]), None
        return None, None
    if ptype == "citation":
        path = os.path.join(SRC_DIR, f"{model_key}.json")
        if os.path.exists(path):
            with open(path) as fh:
                return None, json.load(fh).get(item["item_id"], {}).get(item["set"])
        return None, None
    return None, None


def _pilot_sample(items: list[dict], n_per_set: int) -> list[dict]:
    """N items from each set, spread evenly over domains (round-robin)."""
    out = []
    for which in ("A", "B"):
        by_domain: dict[str, list[dict]] = {}
        for it in items:
            if it["set"] == which:
                by_domain.setdefault(it.get("domain", "unknown"), []).append(it)
        picked, domains = [], sorted(by_domain)
        while len(picked) < n_per_set and any(by_domain[d] for d in domains):
            for d in domains:
                if by_domain[d] and len(picked) < n_per_set:
                    picked.append(by_domain[d].pop(0))
        out += picked
    return out


def run(model_key: str, cfg: dict, system_prompt: str, pilot: int = 0,
        dry_run: bool = False) -> None:
    """pilot=N restricts the run to N items per set (A and B), sampled evenly
    across domains, so the pilot exercises every condition instead of whatever
    happens to come first. dry_run builds the job list and prints the plan and a
    cost estimate WITHOUT calling any model."""
    # a dry run must not need an API key: it plans the work, it does not do it
    prov = None if dry_run else Registry().get(model_key)
    judge = None if dry_run else Judge(cfg)
    bank = load_bank()
    items = load_items(model_key)
    if pilot:
        items = _pilot_sample(items, pilot)
    rng = random.Random(cfg["seed"])

    out = os.path.join(ROOT, "runs", f"study1_{model_key}_{system_prompt}.jsonl")
    already = done_ids(out)
    jobs = []

    for item in items:
        for ptype, spec in cfg["conditions"].items():
            obj, src = _aux(model_key, item, ptype)
            if ptype == "objection" and not obj:
                continue                      # no verified objection for this item
            if ptype == "citation" and not src:
                continue
            identities = spec.get("identities", [None]) if ptype == "identity" else [None]
            for ident in identities:
                jobs.append((item, ptype, "ascending", ident, obj, src))
            # Single-shot arm: each tier alone, to separate intensity from
            # accumulation. Seeded PER ITEM, not from a run-level RNG. A shared
            # RNG is consumed as the job list is built, so a run that skips
            # different cells draws a different subset, and the baseline and its
            # mitigation replays end up with different single-shot items. The
            # ascending arm is unaffected, and it is the arm the paired
            # mitigation comparison uses.
            if spec.get("graded") and random.Random(
                    f"{cfg['seed']}:{item['item_id']}:{ptype}").random() \
                    < cfg["arms"]["single_shot_fraction"]:
                for tier in (1, 2, 3):
                    jobs.append((item, ptype, f"single_shot_{tier}", None, obj, src))

    def one(job) -> None:
        item, ptype, arm, ident, obj, src = job
        try:
            rec = run_conversation(prov, judge, item, ptype, cfg, bank, arm=arm,
                                   identity=ident, system_prompt=system_prompt,
                                   study="study1", objection_text=obj, source_text=src)
        except Exception as exc:  # noqa: BLE001 - never let one item kill the run
            append(out, {"conversation_id": f"ERR:{item['item_id']}:{ptype}:{arm}:{ident}",
                         "error": str(exc)})
            return
        if rec["conversation_id"] in already:
            return
        append(out, rec)

    todo = [j for j in jobs if conversation_id(
        study="study1", model=model_key, item=j[0]["item_id"], ptype=j[1],
        arm=j[2], identity=j[3], system=system_prompt,
        tiers=schedule(j[1], cfg, j[2])) not in already]

    by_type = Counter(j[1] for j in todo)
    n_items = len({j[0]["item_id"] for j in todo})
    print(f"{model_key}/{system_prompt}"
          f"{' [PILOT]' if pilot else ''}{' [DRY RUN]' if dry_run else ''}: "
          f"{len(todo)} conversations over {n_items} items "
          f"({len(jobs) - len(todo)} already done, skipped)")
    print("  by pressure type: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    calls = sum(len(schedule(j[1], cfg, j[2])) + 3 for j in todo)      # +answer,+stability,+probe
    print(f"  ~{calls} model calls, ~{calls * cfg['judging']['self_consistency_n']} judge calls")
    if dry_run:
        print("  dry run - nothing sent. Drop --dry-run to execute.")
        return

    with ThreadPoolExecutor(max_workers=cfg["budget"]["workers"]) as pool:
        list(pool.map(one, todo))
    print(f"written -> {out}")
    if pilot:
        print("  PILOT DONE. Read the actual spend on the provider dashboard, divide by")
        print(f"  {len(todo)} conversations, and multiply by the full-run count from --dry-run.")


if __name__ == "__main__":
    with open(os.path.join(ROOT, "config", "experiment.yaml")) as fh:
        CFG = yaml.safe_load(fh)
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--system", default="baseline",
                    choices=["baseline", "evidence_gate", "generic_warning"])
    ap.add_argument("--pilot", type=int, default=0,
                    help="items per set (A and B) - e.g. --pilot 5 for a costed trial")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and call counts without contacting any model")
    args = ap.parse_args()
    for key in args.models:
        run(key, CFG, args.system, pilot=args.pilot, dry_run=args.dry_run)