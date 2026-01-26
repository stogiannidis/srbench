import re
import glob
import os
import json
import pandas as pd
import argparse
import sys


def normalize_answer(ans):
    try:
        if isinstance(ans, str):
            ans = ans.strip().upper()
            ans = ans.split(".")[0]  # Take text before the first period
    except Exception as e:
        pass
    return ans


def extract_answer_json(text):
    """
    Extracts the answer from a JSON-formatted response.
    Falls back to regex extraction if JSON parsing fails.
    
    Args:
        text (str): The LLM-generated text, expected to contain JSON like {"answer": "A"}
        
    Returns:
        str: The extracted answer in uppercase (e.g., "A", "YES") or result from fallback extraction.
    """
    if pd.isna(text) or text == "":
        return "None"
    
    text = str(text)
    
    # Try to find and parse JSON in the response
    # Look for {"answer": ...} pattern anywhere in the text
    json_pattern = r'\{[^{}]*"answer"\s*:\s*"?([^"}\s]+)"?[^{}]*\}'
    json_match = re.search(json_pattern, text, re.IGNORECASE)
    
    if json_match:
        # Try to extract the answer value
        answer = json_match.group(1).strip().upper()
        if answer:
            return answer
    
    # Try to parse the entire text or find embedded JSON
    try:
        # Look for JSON object in the text
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            if isinstance(data, dict) and 'answer' in data:
                return str(data['answer']).strip().upper()
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback to standard extraction
    return extract_answer(text)


def extract_answer(text):
    """
    Extracts the final answer from an LLM's output, supporting single letters,
    specific words, Markdown bolding, and mixed formats (e.g., "C. Left").

    Args:
        text (str): The LLM-generated text.

    Returns:
        str: The extracted answer in uppercase (e.g., "A", "YES") or "None" if not found.
    """
    if pd.isna(text) or text == "":
        return "None"

    text = str(text)

    # --- Step 1: Clean up the input text ---
    prefixes_to_remove = ["Assistant", "ASSISTANT", "[INST]", "assistant"]
    first_prefix_pos = len(text)
    for prefix in prefixes_to_remove:
        pos = text.find(prefix)
        if pos != -1:
            first_prefix_pos = min(first_prefix_pos, pos)

    if first_prefix_pos != len(text):
        text = text[first_prefix_pos:]

    # --- Step 2: Define answer patterns ---
    word_answers = ["yes", "no", "left", "right", "back", "front", "center"]
    core_pattern = r"\b(" + "|".join(word_answers) + r"|[A-Z])\b"
    answer_pattern = r"(?:\*\*)?" + core_pattern + r"(?:\*\*)?"

    # --- Step 3: Check for answers in different formats, from most to least specific ---

    # A. Check for answers in curly brackets, e.g., {**A**}
    curly_pattern = (
        r"\{" + r"(?:\*\*)?(" + "|".join(word_answers) + r"|[A-Z])(?:\*\*)?" + r"\}"
    )
    curly_match = re.search(curly_pattern, text, re.IGNORECASE)
    if curly_match:
        return curly_match.group(1).upper()

    # B. Check for mixed format like "C. Left" and prioritize the letter.
    mixed_pattern = r"\b([A-Z])(?:\.|:|\))\s*(?:" + "|".join(word_answers) + r")\b"
    mixed_match = re.search(mixed_pattern, text, re.IGNORECASE)
    if mixed_match:
        return mixed_match.group(1).upper()

    # C. Check for phrases that typically precede or follow the answer
    before_phrases = [
        "the answer is",
        "i think it's",
        "i choose",
        "i'll go with",
        "it's",
        "the correct choice is",
        "my answer is",
        "i believe it's",
        "i select",
        "the best answer is",
    ]
    after_phrases = [
        "is the answer",
        "is correct",
        "is the correct choice",
        "is right",
        "is the best answer",
        "is the right choice",
    ]

    before_pattern = (
        r"(?:"
        + "|".join(re.escape(p) for p in before_phrases)
        + r")\s*"
        + answer_pattern
    )
    after_pattern = (
        answer_pattern
        + r"\s*(?:"
        + "|".join(re.escape(p) for p in after_phrases)
        + r")"
    )

    before_regex = re.compile(before_pattern, re.IGNORECASE)
    after_regex = re.compile(after_pattern, re.IGNORECASE)

    matches = list(before_regex.finditer(text)) + list(after_regex.finditer(text))
    if matches:
        matches.sort(key=lambda m: m.start())
        return matches[-1].group(1).upper()

    # D. Check for a direct answer format: "**A**.", "**Yes**:", etc.
    direct_match = re.search(
        answer_pattern + r"(?:\.|:|\))?(?:\s|$)", text, re.IGNORECASE
    )
    if direct_match:
        return direct_match.group(1).upper()

    # E. Fallback: find the last standalone answer word/letter in the text
    fallback_matches = re.findall(answer_pattern, text, re.IGNORECASE)
    if fallback_matches:
        return fallback_matches[-1].upper()

    return "None"


def exact_match(pred, gold):
    """Compare normalized predicted and gold answers"""
    return normalize_answer(pred) == normalize_answer(gold)


def process_csv_files(
    pattern, response_col, correct_col, split_col=None, output_file=None
):
    """
    Process multiple CSV files and calculate accuracy per split.

    Args:
        pattern (str): Glob pattern for CSV files
        response_col (str): Name of column containing model responses
        correct_col (str): Name of column containing correct answers
        split_col (str, optional): Name of column containing splits
        output_file (str, optional): Path to save results CSV

    Returns:
        pd.DataFrame: DataFrame with results per model and split
    """
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None

    print(f"Found {len(files)} files to process")

    # Dictionary to store results for all models
    results = {}
    detailed_results = []

    for file in files:
        print(f"Processing: {file}")

        try:
            df = pd.read_csv(file)
            print(f"  Loaded {len(df)} rows")
        except Exception as e:
            print(f"  Error reading {file}: {e}")
            continue

        # Extract model name from filename
        model_name = os.path.basename(file).replace(".csv", "")

        # Check if required columns exist
        if response_col not in df.columns:
            print(f"  Warning: Column '{response_col}' not found in {file}")
            continue
        if correct_col not in df.columns:
            print(f"  Warning: Column '{correct_col}' not found in {file}")
            continue
        if split_col and split_col not in df.columns:
            print(f"  Warning: Column '{split_col}' not found in {file}")
            continue

        # Extract answers from responses
        df["extracted_answer"] = (
            df[response_col].fillna("").astype(str).apply(extract_answer)
        )

        # Calculate correctness using exact match with extracted answers
        try:
            df["correct"] = df.apply(
                lambda row: exact_match(row["extracted_answer"], row[correct_col]),
                axis=1,
            )
        except Exception as e:
            print(f"  Error calculating correctness for {file}: {e}")
            continue

        # Calculate overall accuracy
        overall_accuracy = df["correct"].mean()

        # Initialize results for this model
        results[model_name] = {"overall": overall_accuracy}

        # Calculate accuracy per split if split column exists
        if split_col and split_col in df.columns:
            try:
                accuracy_per_split = df.groupby(split_col)["correct"].mean()
                results[model_name].update(accuracy_per_split.to_dict())

                print(f"  Accuracy per split:")
                for split, acc in accuracy_per_split.items():
                    print(f"    {split}: {acc:.4f} ({acc * 100:.2f}%)")

            except Exception as e:
                print(f"  Error calculating accuracy per split for {file}: {e}")

        print(
            f"  Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)"
        )

        # Store detailed results for this file
        file_details = {
            "model": model_name,
            "total_questions": len(df),
            "correct_answers": int(df["correct"].sum()),
            "extraction_failed": int((df["extracted_answer"] == "None").sum()),
            "overall_accuracy": overall_accuracy,
        }
        detailed_results.append(file_details)

        print()

    if not results:
        print("No valid results found")
        return None

    # Create DataFrame with models as rows and splits as columns
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_index()
    results_df.index.name = "model"

    # Round all values to 4 decimal places
    results_df = results_df.round(4)

    # Save results
    if output_file is None:
        output_file = "model_accuracy_by_split.csv"

    results_df.to_csv(output_file)
    print(f"Results saved to {output_file}")

    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_output = output_file.replace(".csv", "_detailed.csv")
    detailed_df.to_csv(detailed_output, index=False)
    print(f"Detailed results saved to {detailed_output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(results_df)
    print("=" * 60)

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate accuracy from multiple CSV files"
    )
    parser.add_argument(
        "--input-pattern",
        "-i",
        required=True,
        help='Glob pattern for CSV files (e.g., "data/*.csv")'
    )
    parser.add_argument(
        "--response-col",
        "-r",
        required=True,
        help="Name of column containing model responses",
    )
    parser.add_argument(
        "--correct-col",
        "-c",
        required=True,
        help="Name of column containing correct answers",
    )
    parser.add_argument(
        "--split-col", "-s", help="Name of column containing splits (optional)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save results CSV (default: model_accuracy_by_split.csv)",
    )

    args = parser.parse_args()

    # Process files
    results_df = process_csv_files(
        args.pattern, args.response_col, args.correct_col, args.split_col, args.output
    )

    if results_df is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
