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