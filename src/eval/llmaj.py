#!/usr/bin/env python3
"""
Judge if two answers are equivalent (binary 0/1) with robust heuristics.
- Pre-check: if both answers have choice labels (A-E), compare labels only.
- Else: compare normalized text.
- Optional: LLM fallback (disabled by default unless model provided).
Writes result to a 'score' column for each row.
"""
import argparse
import pandas as pd
import os
import sys
import logging
import glob
import re
from typing import List, Optional
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_pipeline(model_name: Optional[str] = None):
    """Create a text generation pipeline if model_name is provided; else return None."""
    if not model_name:
        return None
    try:
        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            return_full_text=False,
            max_length=1000,
        )
        return pipe
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        return None

def load_csv_files(directory: str) -> List[str]:
    """Find all CSV files in directory."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")
    return sorted(csv_files)

# --- Heuristics to reduce misclassification ---
CHOICE_RE = re.compile(r"^\s*([A-Ea-e])\s*([\).: -]|$)")

def extract_choice_label(text: str) -> Optional[str]:
    """Extract leading multiple-choice label (A-E) if present."""
    if not isinstance(text, str):
        return None
    m = CHOICE_RE.match(text)
    if m:
        return m.group(1).upper()
    # Also handle compact forms like "B.Center" or "A)Left" with no space
    if text and len(text) >= 2 and text[0].upper() in "ABCDE" and text[1] in "):. -":
        return text[0].upper()
    return None

def normalize_simple(text: str) -> str:
    """Lowercase, strip punctuation/extra spaces for simple equality check."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"[\W_]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", t)

def safe_parse_binary(output_text: str) -> str:
    """Return '0' or '1' from model output; default to '0' on ambiguity."""
    if not isinstance(output_text, str):
        return "0"
    for ch in output_text.strip():
        if ch in {"0", "1"}:
            return ch
    return "0"

def llm_judge(pipe, a1: str, a2: str) -> str:
    """Optional LLM judge; returns '0' or '1'."""
    if pipe is None:
        return "0"  # default to conservative mismatch
    
    # Build a strict single-string prompt for text-generation models
    prompt = (
        "You are an answer matching recogniser. Given an LLM response and the gold answer, decide if the LL's response is the same as the gold answer.\n\n"
        "Rules:\n- If the LLM's answers results to the same output as the gold answer -> output 1.\n- If the answers differ in any way (wording, choice letter, or semantics) -> output 0.\n\n"
        "Output exactly one character: 0 or 1.\n\n"
        f"LLM answer: {a1}\nGold Answer: {a2}\nAnswer:"
    )
    
    try:
        out = pipe(prompt, max_new_tokens=1, do_sample=False, temperature=0.0)
        text = out[0].get("generated_text", "") if isinstance(out, list) else ""
        return safe_parse_binary(text)
    except Exception as e:
        logger.warning(f"LLM judge failed, defaulting to 0: {e}")
        return "0"

def process_csv_file(filepath: str, normalizer, args) -> bool:
    """Process a single CSV file."""
    try:
        logger.info(f"Processing: {os.path.basename(filepath)}")
        df = pd.read_csv(filepath)
        
        if args.response_column not in df.columns:
            logger.warning(f"Column '{args.response_column}' not found in {filepath}")
            return False
        if "question" not in df.columns or "gold answer" not in df.columns:
            logger.warning(f"Required columns 'question' and/or 'gold answer' not found in {filepath}")
            return False
        
        scores = []
        for _, row in df.iterrows():
            question = str(row.get("question", ""))
            a1 = str(row.get(args.response_column, ""))
            a2 = str(row.get("gold answer", ""))
            
            # # Heuristic 1: compare explicit choice labels first
            # a1_label = extract_choice_label(a1)
            # a2_label = extract_choice_label(a2)
            # if a1_label and a2_label:
            #     scores.append("1" if a1_label == a2_label else "0")
            #     continue
            
            # # Heuristic 2: simple normalized equality
            # n1, n2 = normalize_simple(a1), normalize_simple(a2)
            # if n1 and n1 == n2:
            #     scores.append("1")
            #     continue
            
            # Fallback to LLM judge (optional)
            scores.append(llm_judge(normalizer, a1, a2))
        
        df["score"] = scores
        df.to_csv(filepath, index=False)
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {filepath}: {e}", exc_info=True)
        return False

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Judge equivalence of answers in CSV files and write a binary score column.")
    parser.add_argument("--input_directory", required=True, help="Directory containing CSV files")
    parser.add_argument("--response-column", dest="response_column", default="raw_response", help="Response column name (default: raw_response)")
    parser.add_argument("--model", dest="model_name", default=None, help="Optional HF model name for LLM fallback (default: disabled)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if not os.path.isdir(args.input_directory):
        logger.error(f"Directory not found: {args.input_directory}")
        sys.exit(1)
    
    csv_files = load_csv_files(args.input_directory)
    if not csv_files:
        logger.error("No CSV files found")
        sys.exit(1)
        
    normalizer = create_pipeline(model_name=args.model_name)
    
    successful = 0
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"[{i}/{len(csv_files)}] {os.path.basename(csv_file)}")
        if process_csv_file(csv_file, normalizer, args):
            successful += 1
    
    logger.info(f"Complete: {successful}/{len(csv_files)} files processed successfully")

if __name__ == "__main__":
    main()