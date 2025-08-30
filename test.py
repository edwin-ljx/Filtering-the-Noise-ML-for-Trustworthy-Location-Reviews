import os
import csv
import sys
from typing import Tuple, Optional, List

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# Model & Prompt
# -------------------------
model = OllamaLLM(model="llama3.2")

template = """
You are an expert in identifying trustworthy and policy-compliant location reviews.

Location Being Reviewed:
{location}

Review to Evaluate:
{review}

Policies to Enforce (sentiment-neutral):
    1. No Advertisement: Reviews should not contain promotional content, discount offers, or links.
    2. No Irrelevant Content: Reviews should focus on the experience at the specified location, not unrelated matters.
    3. No Rant Without Visit: Complaints or praise must be based on an actual visit/experience; pure speculation, hearsay, or second-hand rants should be flagged.

Important: Negative or strongly critical reviews are allowed and should not be flagged solely due to sentiment. Only flag if a policy is violated.

Instructions:
    • Determine whether the review is Valid or Flagged.
    • If flagged, choose the single strongest policy violated (Primary Violation).
    • Provide a brief explanation (1 to 2 sentences).

Output Format:
• Decision: Valid / Flagged
• Primary Violation (if flagged): [Policy Name]
• Explanation: [Short reasoning]
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# -------------------------
# Helpers
# -------------------------

def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")

def _guess_columns(headers: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to auto-detect location/review columns using common names.
    Returns (location_col, review_col) or (None, None) if not found.
    """
    norm_map = {_normalize(h): h for h in headers}

    # Common variants
    loc_candidates = [
        "location", "place", "venue", "restaurant", "store", "hotel", "site", "spot"
    ]
    rev_candidates = [
        "review", "text", "comment", "feedback", "content", "body", "ratingtext"
    ]

    found_loc = next((norm_map[n] for n in loc_candidates if n in norm_map), None)
    found_rev = next((norm_map[n] for n in rev_candidates if n in norm_map), None)
    return found_loc, found_rev

def _parse_model_output(output_text: str) -> Tuple[str, str, str]:
    """
    Best-effort parser for the model's three-line output format.
    Returns (Decision, Primary Violation, Explanation).
    If a field isn't found, returns empty string for that field.
    """
    lines = [l.strip() for l in output_text.splitlines() if l.strip()]
    decision, violation, explanation = "", "", ""

    for ln in lines:
        low = ln.lower()
        if "decision:" in low and not decision:
            decision = ln.split(":", 1)[1].strip()
        elif "primary violation" in low and not violation:
            violation = ln.split(":", 1)[1].strip()
        elif "explanation:" in low and not explanation:
            explanation = ln.split(":", 1)[1].strip()

    return decision, violation, explanation

def evaluate(location: str, review: str) -> Tuple[str, str, str, str]:
    """
    Run the chain and return a tuple:
    (Decision, Primary Violation, Explanation, RawOutput)
    """
    res = chain.invoke({"location": location.strip(), "review": review.strip()})
    # res may be a string or an object with .content depending on your LangChain version
    text = getattr(res, "content", res) if not isinstance(res, str) else res
    decision, violation, explanation = _parse_model_output(text)
    return decision, violation, explanation, text

def process_csv(path: str) -> str:
    """
    Evaluate all rows in a CSV file and write an output CSV with appended columns.
    Returns the output CSV path.
    """
    if not os.path.isfile(path):
        print(f"[Error] File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers:
            print("[Error] No headers found in CSV.")
            sys.exit(1)

        loc_col, rev_col = _guess_columns(headers)

        if not loc_col or not rev_col:
            print("\nCould not auto-detect columns.")
            print("CSV headers:", headers)
            # Ask user which columns to use (keeps the script flexible)
            loc_col = input("Enter the column name for LOCATION: ").strip()
            rev_col = input("Enter the column name for REVIEW: ").strip()
            if loc_col not in headers or rev_col not in headers:
                print("[Error] Provided column names are not in CSV headers.")
                sys.exit(1)

        # Prepare output path
        base, ext = os.path.splitext(path)
        out_path = f"{base}_evaluated{ext}"

        out_headers = headers + ["Decision", "Primary Violation", "Explanation"]
        # If you also want the raw LLM output for debugging, append "RawOutput"
        include_raw = False
        if include_raw and "RawOutput" not in out_headers:
            out_headers.append("RawOutput")

        with open(out_path, "w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=out_headers)
            writer.writeheader()

            row_idx = 0
            for row in reader:
                row_idx += 1
                location = (row.get(loc_col) or "").strip()
                review = (row.get(rev_col) or "").strip()

                if not review:
                    # Skip empty review rows, but still write row unchanged
                    row["Decision"] = ""
                    row["Primary Violation"] = ""
                    row["Explanation"] = ""
                    if include_raw:
                        row["RawOutput"] = ""
                    writer.writerow(row)
                    continue

                decision, violation, explanation, raw = evaluate(location, review)
                row["Decision"] = decision
                row["Primary Violation"] = violation
                row["Explanation"] = explanation
                if include_raw:
                    row["RawOutput"] = raw
                writer.writerow(row)

                print(f"Processed row {row_idx}: Decision={decision or '—'}")

    return out_path

# -------------------------
# Main loop with mode select
# -------------------------
if __name__ == "__main__":
    while True:
        print("\n\n-----------------------")
        mode = input("Mode: (i) individual review, (f) CSV file, (q) quit: ").strip().lower()
        if mode == "q":
            break

        if mode == "i":
            location_input = input("Enter the location being reviewed (q to quit): ").strip()
            if location_input.lower() == "q":
                break
            review_input = input("Enter your review: ").strip()
            print("\nEvaluating...\n")
            decision, violation, explanation, _ = evaluate(location_input, review_input)
            print(f"Decision: {decision or '—'}")
            if violation:
                print(f"Primary Violation: {violation}")
            print(f"Explanation: {explanation or '—'}")

        elif mode == "f":
            csv_path = input("Path to CSV file: ").strip().strip('"').strip("'")
            print("\nEvaluating CSV rows...\n")
            out_csv = process_csv(csv_path)
            print(f"\nDone. Output written to: {out_csv}")
        else:
            print("Please enter 'i' for individual, 'f' for file, or 'q' to quit.")
