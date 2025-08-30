#!/usr/bin/env python3
# interactive_evaluate_reviews.py

import csv
import os

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# --- Canonical mappings ---
CANONICAL_DECISIONS = {"valid": "Valid", "flagged": "Flagged"}
CANONICAL_VIOLATIONS = {
    "no advertisement": "No Advertisement",
    "no irrelevant content": "No Irrelevant Content",
    "no rant without visit": "No Rant Without Visit",
    "none": "None",
}
VIOLATION_ALIASES = {
    "advertising": "No Advertisement",
    "advertisement": "No Advertisement",
    "ads": "No Advertisement",
    "promo": "No Advertisement",
    "promotion": "No Advertisement",
    "irrelevant": "No Irrelevant Content",
    "off-topic": "No Irrelevant Content",
    "rant without visit": "No Rant Without Visit",
    "speculation": "No Rant Without Visit",
    "hearsay": "No Rant Without Visit",
    "none": "None",
    "": "None",
}

def norm(s: Optional[str]) -> str:
    return " ".join(s.strip().lower().split()) if s else ""

def canonical_decision(s: str) -> str:
    key = norm(s)
    if key.startswith("valid"):
        return "Valid"
    if key.startswith("flag"):
        return "Flagged"
    return CANONICAL_DECISIONS.get(key, s.strip())

def canonical_violation(s: str) -> str:
    key = norm(s)
    if key in CANONICAL_VIOLATIONS:
        return CANONICAL_VIOLATIONS[key]
    if key in VIOLATION_ALIASES:
        return VIOLATION_ALIASES[key]
    if "advert" in key or "promo" in key:
        return "No Advertisement"
    if "irrelev" in key or "offtopic" in key:
        return "No Irrelevant Content"
    if "visit" in key or "hearsay" in key or "speculat" in key or "rant" in key:
        return "No Rant Without Visit"
    return "None" if key == "" else s.strip()

# --- Metrics helpers ---
def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1

# --- Interactive flow ---
def ask(msg: str, default: Optional[str] = None) -> str:
    if default:
        val = input(f"{msg} [{default}]: ").strip()
        return val if val else default
    return input(f"{msg}: ").strip()

def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def evaluate(rows: List[Dict[str,str]], 
             pred_dec_col: str, pred_vio_col: str, 
             gt_dec_col: str, gt_vio_col: str, 
             id_col: Optional[str]):

    total = len(rows)
    correct_dec, correct_vio = 0, 0
    mismatches = []

    for r in rows:
        pid = r.get(id_col, "") if id_col else ""
        pred_dec = canonical_decision(r.get(pred_dec_col, ""))
        gt_dec   = canonical_decision(r.get(gt_dec_col, ""))

        pred_vio = canonical_violation(r.get(pred_vio_col, ""))
        gt_vio   = canonical_violation(r.get(gt_vio_col, ""))

        # normalize: Valid -> None violation
        if pred_dec == "Valid": pred_vio = "None"
        if gt_dec == "Valid": gt_vio = "None"

        if pred_dec == gt_dec:
            correct_dec += 1
        if pred_vio == gt_vio:
            correct_vio += 1
        else:
            mismatches.append({
                "id": pid,
                "pred_decision": pred_dec,
                "gt_decision": gt_dec,
                "pred_violation": pred_vio,
                "gt_violation": gt_vio,
                "review": r.get("review", r.get("Review", "")),
            })

    print("\n=== Results ===")
    print(f"Total rows: {total}")
    print(f"Decision Accuracy: {correct_dec}/{total} = {correct_dec/total:.2%}")
    print(f"Violation Accuracy: {correct_vio}/{total} = {correct_vio/total:.2%}")

    # Save mismatches
    out_file = "mismatches.csv"
    with open(out_file, "w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=mismatches[0].keys())
        writer.writeheader()
        writer.writerows(mismatches)
    print(f"\nSaved mismatch report: {out_file}")

def main():
    print("=== Review Evaluation Script ===\n")
    file_path = ask("Enter path to CSV file")
    if not os.path.isfile(file_path):
        print("File not found.")
        return

    rows = load_csv(file_path)
    headers = list(rows[0].keys())
    print("\nCSV headers detected:", headers)

    # Prompt for columns
    pred_dec_col = ask("Prediction Decision column", "Decision")
    pred_vio_col = ask("Prediction Violation column", "Primary Violation")
    gt_dec_col   = ask("Ground Truth Decision column", "GT_Decision")
    gt_vio_col   = ask("Ground Truth Violation column", "GT_Violation")
    id_col       = ask("ID column (optional, Enter to skip)") or None

    evaluate(rows, pred_dec_col, pred_vio_col, gt_dec_col, gt_vio_col, id_col)

if __name__ == "__main__":
    main()
