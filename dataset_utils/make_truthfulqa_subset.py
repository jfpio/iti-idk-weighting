#!/usr/bin/env python3
"""
Subset TruthfulQA to a fast-eval split with:
  - proportionate per-category sampling
  - target global reference-IDK rate (default 10% ± 2pp)

Usage:
  python make_truthfulqa_subset.py \
      --input ../datasets/TruthfulQA_original.csv \
      --output subset_eval_100.csv \
      --n 100 \
      --target-idk 0.10 \
      --tol 0.02 \
      --category-col Category \
      --id-col QuestionID

Notes
- "reference-IDK" means the question has at least one reference answer that is a non-assertive refusal (e.g., "I have no comment / I don't know").
- IDK detection uses exact template matching with comprehensive IDK patterns but you can pass extra phrases via --extra-idk-phrases-file (one per line).
- The script tries to auto-detect answer columns. You can override via --answer-cols Answer1 Answer2 ... or a single --answers-col that contains a concatenated string of answers.
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# Import our improved IDK detection system
try:
    from .idk_rephraser import IDKRephraser
except ImportError:
    # Fallback for when running as a script
    from idk_rephraser import IDKRephraser


# ---------- IDK detection utilities ----------
def get_idk_patterns(extra_phrases=None):
    """
    Get all IDK detection patterns using our improved IDKRephraser system.
    Returns a set of lowercase patterns for exact template matching.
    """
    rephraser = IDKRephraser()
    patterns = set(pattern.lower().strip() for pattern in rephraser.get_all_detection_patterns())
    
    # Add any extra phrases if provided
    if extra_phrases:
        for phrase in extra_phrases:
            if phrase and phrase.strip():
                patterns.add(phrase.strip().lower())
    
    return patterns

def normalize_text(s: str) -> str:
    """Normalize text for exact template matching."""
    if not isinstance(s, str):
        return ""
    # Lowercase, normalize unicode quotes/dashes lightly, and strip
    s = s.strip().lower()
    s = s.replace("'", "'").replace("'", "'").replace("–", "-").replace("—", "-")
    # Collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s


# ---------- Answer extraction ----------
def autodetect_answer_columns(df: pd.DataFrame):
    # Prefer columns that look like reference answers
    cols = [c for c in df.columns if re.search(r"(answer|ref|reference)", c, flags=re.IGNORECASE)]
    # Exclude obvious non-text columns
    bad = {"AnswerType", "AnswerIndex", "IsCorrect", "TruthLabel"}
    cols = [c for c in cols if c not in bad]
    # Heuristic: keep only string-like columns
    str_cols = []
    for c in cols:
        # sample a few to see if mostly strings
        sample = df[c].dropna().head(20)
        if len(sample) == 0:
            continue
        if sample.apply(lambda x: isinstance(x, str)).mean() >= 0.6:
            str_cols.append(c)
    return str_cols

def extract_answers(row, answer_cols=None, answers_col=None, answers_sep="||"):
    answers = []
    if answer_cols:
        for c in answer_cols:
            val = row.get(c, None)
            if isinstance(val, str) and val.strip():
                answers.append(val)
    elif answers_col:
        val = row.get(answers_col, None)
        if isinstance(val, str) and val.strip():
            parts = [p.strip() for p in val.split(answers_sep)]
            answers.extend([p for p in parts if p])
    else:
        # Fallback: try autodetect
        # (This path should be avoided in production; better pass explicit cols)
        pass
    return answers


# ---------- Sampling helpers ----------
def proportional_allocation(counts_by_cat: dict, N: int):
    total = sum(counts_by_cat.values())
    # Ideal fractional allocation
    ideal = {c: N * counts_by_cat[c] / total for c in counts_by_cat}
    # Floor, then distribute remainder by largest fractional part
    alloc = {c: int(math.floor(ideal[c])) for c in ideal}
    remainder = N - sum(alloc.values())
    if remainder > 0:
        frac_parts = sorted(((ideal[c] - alloc[c], c) for c in alloc), reverse=True)
        i = 0
        while remainder > 0 and i < len(frac_parts):
            c = frac_parts[i][1]
            alloc[c] += 1
            remainder -= 1
            i += 1
    elif remainder < 0:
        # Remove from smallest fractional parts (shouldn't happen often)
        frac_parts = sorted(((ideal[c] - alloc[c], c) for c in alloc))
        i = 0
        while remainder < 0 and i < len(frac_parts):
            c = frac_parts[i][1]
            if alloc[c] > 0:
                alloc[c] -= 1
                remainder += 1
            i += 1
    return alloc

def clamp_alloc_to_available(alloc: dict, available_by_cat: dict):
    alloc = alloc.copy()
    deficit = 0
    for c, k in list(alloc.items()):
        cap = available_by_cat.get(c, 0)
        if k > cap:
            deficit += (k - cap)
            alloc[c] = cap
    if deficit == 0:
        return alloc
    # Redistribute deficit among categories that still have spare capacity
    # based on their spare capacity size
    spare = {c: available_by_cat[c] - alloc[c] for c in alloc}
    while deficit > 0:
        # pick the category with the largest spare
        cand = sorted(spare.items(), key=lambda x: x[1], reverse=True)
        progressed = False
        for c, s in cand:
            if s <= 0:
                continue
            alloc[c] += 1
            spare[c] -= 1
            deficit -= 1
            progressed = True
            if deficit == 0:
                break
        if not progressed:
            break  # cannot satisfy further
    return alloc

def greedy_sample_with_idk_target(
    df,
    id_col,
    category_col,
    idk_flag_col,
    alloc_by_cat,
    target_idk_count,
    rng: np.random.Generator,
):
    """
    Sample without replacement per category to match allocation and approximately meet total IDK target.
    Strategy:
      1) Split each category pool into IDK vs non-IDK.
      2) Propose per-category IDK picks proportional to available IDK and limited by alloc.
      3) Adjust to hit the global target (greedy).
      4) Sample per category accordingly; borrow if shortages occur.
    """
    # Build pools
    pools = {}
    total_avail_idk = 0
    for c, k in alloc_by_cat.items():
        cat_df = df[df[category_col] == c]
        idk_pool = cat_df[cat_df[idk_flag_col]].index.tolist()
        non_pool = cat_df[~cat_df[idk_flag_col]].index.tolist()
        pools[c] = {"idk": idk_pool, "non": non_pool, "alloc": int(k)}
        total_avail_idk += len(idk_pool)
    target_idk_count = min(target_idk_count, total_avail_idk)

    # Initial proportional idk allocation
    avail_idk_by_cat = {c: len(pools[c]["idk"]) for c in pools}
    total_idk_avail = sum(avail_idk_by_cat.values()) or 1
    idk_alloc = {}
    for c, p in pools.items():
        k = p["alloc"]
        prop = target_idk_count * (avail_idk_by_cat[c] / total_idk_avail)
        idk_alloc[c] = min(int(round(prop)), k, avail_idk_by_cat[c])

    # Clip to per-category alloc and adjust total to match target
    def total_idk(a): return sum(a.values())
    # Reduce if overshoot
    while total_idk(idk_alloc) > target_idk_count:
        # take one from the category with largest idk_alloc
        c = max(idk_alloc, key=lambda x: idk_alloc[x])
        if idk_alloc[c] > 0:
            idk_alloc[c] -= 1
        else:
            break
    # Increase if undershoot
    tries = 0
    while total_idk(idk_alloc) < target_idk_count and tries < 5 * len(idk_alloc):
        # give one to category with biggest remaining room (min(k - idk_alloc, avail_idk - idk_alloc))
        c = max(
            idk_alloc,
            key=lambda x: min(
                pools[x]["alloc"] - idk_alloc[x],
                avail_idk_by_cat[x] - idk_alloc[x],
            ),
        )
        room = min(pools[c]["alloc"] - idk_alloc[c], avail_idk_by_cat[c] - idk_alloc[c])
        if room > 0:
            idk_alloc[c] += 1
        tries += 1

    # Now sample per category
    chosen_indices = []
    for c, p in pools.items():
        k = p["alloc"]
        k_idk = min(idk_alloc[c], len(p["idk"]), k)
        # sample idk
        if k_idk > 0:
            chosen_indices.extend(rng.choice(p["idk"], size=k_idk, replace=False).tolist())
        k_non = k - k_idk
        if k_non > 0:
            non_pool = [i for i in p["non"] if i not in chosen_indices]
            # If not enough non-idk in this category, borrow from idk (already maxed) or accept shortfall
            take = min(k_non, len(non_pool))
            if take > 0:
                chosen_indices.extend(rng.choice(non_pool, size=take, replace=False).tolist())

    # If we are short due to borrowing issues, fill from global remaining pool
    if len(chosen_indices) < sum(alloc_by_cat.values()):
        needed = sum(alloc_by_cat.values()) - len(chosen_indices)
        remaining = [i for i in df.index if i not in chosen_indices]
        if needed > 0 and len(remaining) >= needed:
            chosen_indices.extend(rng.choice(remaining, size=needed, replace=False).tolist())

    return sorted(set(chosen_indices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to TruthfulQA original CSV")
    ap.add_argument("--output", required=True, help="Path to save subset CSV")
    ap.add_argument("--n", type=int, default=100, help="Subset size")
    ap.add_argument("--target-idk", type=float, default=0.10, help="Target IDK rate (0..1)")
    ap.add_argument("--tol", type=float, default=0.02, help="Acceptable absolute deviation for IDK rate")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--id-col", default="QuestionID", help="Question ID column")
    ap.add_argument("--category-col", default="Category", help="Category column")

    # Answer columns (choose one mode)
    ap.add_argument("--answer-cols", nargs="*", default=None,
                    help="List of columns that each contain a reference answer text")
    ap.add_argument("--answers-col", default=None,
                    help="Single column that contains concatenated answers (use --answers-sep)")
    ap.add_argument("--answers-sep", default=";",
                    help="Separator when using --answers-col")

    ap.add_argument("--extra-idk-phrases-file", default=None,
                    help="Optional text file with extra IDK phrases (one per line)")
    ap.add_argument("--stats-json", default=None,
                    help="Optional path to write a small JSON with subset stats")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)
    if args.id_col not in df.columns:
        # If missing, fallback to index as QuestionID
        df[args.id_col] = np.arange(len(df), dtype=int)

    # Auto-detect answers if not specified
    answer_cols = args.answer_cols
    answers_col = args.answers_col
    
    if (not answer_cols) and (not answers_col):
        auto = autodetect_answer_columns(df)
        if auto:
            # Special handling for TruthfulQA format
            # If "Correct Answers" is found and contains semicolons, use concatenated mode
            if "Correct Answers" in auto:
                # Sample a few rows to check if they contain semicolons (indicating concatenated answers)
                sample_vals = df["Correct Answers"].dropna().head(10)
                if any(";" in str(val) for val in sample_vals):
                    answers_col = "Correct Answers"
                    print(f"[INFO] Detected TruthfulQA format: using '{answers_col}' with separator '{args.answers_sep}'")
                else:
                    answer_cols = auto[:6]
            else:
                # Default: use first 6 candidate columns as individual answer columns
                answer_cols = auto[:6]
        
        if not answer_cols and not answers_col:
            raise ValueError(
                "Could not autodetect answer columns. "
                "Pass --answer-cols Answer1 Answer2 ... OR --answers-col answers"
            )

    # Load extra phrases if provided
    extra_phrases = None
    if args.extra_idk_phrases_file and os.path.exists(args.extra_idk_phrases_file):
        with open(args.extra_idk_phrases_file, "r", encoding="utf-8") as f:
            extra_phrases = [line.strip() for line in f if line.strip()]

    # Get IDK patterns using our improved detection system
    idk_patterns = get_idk_patterns(extra_phrases)

    # Compute is_idk_ref per question (reference-side) using exact template matching
    def row_is_idk(row):
        answers = extract_answers(row, answer_cols=answer_cols,
                                  answers_col=answers_col,
                                  answers_sep=args.answers_sep)
        for a in answers:
            norm = normalize_text(a)
            if norm in idk_patterns:
                return True
        return False

    df = df.copy()
    df["_is_idk_ref"] = df.apply(row_is_idk, axis=1)

    # Check required category col
    if args.category_col not in df.columns:
        raise ValueError(f"Category column '{args.category_col}' not found in CSV.")

    # Proportionate allocation
    counts_by_cat = Counter(df[args.category_col])
    alloc = proportional_allocation(counts_by_cat, args.n)
    # Clamp to available per category
    available_by_cat = {c: int((df[args.category_col] == c).sum()) for c in counts_by_cat}
    alloc = clamp_alloc_to_available(alloc, available_by_cat)

    # Compute target IDK count
    target_idk_count = int(round(args.n * args.target_idk))

    # Sample
    chosen_idx = greedy_sample_with_idk_target(
        df=df,
        id_col=args.id_col,
        category_col=args.category_col,
        idk_flag_col="_is_idk_ref",
        alloc_by_cat=alloc,
        target_idk_count=target_idk_count,
        rng=rng,
    )
    subset = df.loc[chosen_idx].copy()

    # Evaluate achieved stats
    achieved_idk_rate = float(subset["_is_idk_ref"].mean()) if len(subset) else 0.0
    within_tol = abs(achieved_idk_rate - args.target_idk) <= args.tol

    # Save subset
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    # Keep original columns, add helper columns at the end
    out_cols = list(df.columns)
    # Ensure helper columns appear at the end if they weren't in original
    if "_is_idk_ref" not in out_cols:
        out_cols = list(df.columns) + ["_is_idk_ref"]
    subset.to_csv(args.output, index=False, columns=out_cols)

    # Stats JSON
    stats = {
        "input_path": os.path.abspath(args.input),
        "output_path": os.path.abspath(args.output),
        "n": int(len(subset)),
        "target_idk": args.target_idk,
        "achieved_idk": round(achieved_idk_rate, 4),
        "within_tolerance": within_tol,
        "tolerance": args.tol,
        "category_counts_full": dict(sorted(counts_by_cat.items())),
        "category_counts_subset": dict(sorted(Counter(subset[args.category_col]).items())),
        "idk_counts_subset": {
            "idk_true": int(subset["_is_idk_ref"].sum()),
            "idk_false": int((~subset["_is_idk_ref"]).sum()),
        },
        "seed": args.seed,
        "answer_cols_used": answer_cols if answer_cols else [answers_col],
        "id_col": args.id_col,
        "category_col": args.category_col,
    }
    if args.stats_json:
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    # Console summary
    print(f"[OK] Wrote subset to: {args.output}")
    print(f"    Subset size: {len(subset)} | Target IDK: {args.target_idk:.3f} | Achieved: {achieved_idk_rate:.3f} | Within tol: {within_tol}")
    print("    Category counts (subset):")
    for c, k in sorted(Counter(subset[args.category_col]).items(), key=lambda x: x[0]):
        print(f"      - {c}: {k}")
    if args.stats_json:
        print(f"    Stats JSON: {args.stats_json}")


if __name__ == "__main__":
    main()
