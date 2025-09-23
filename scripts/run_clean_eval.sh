#!/bin/bash

# Simple bash script to run clean_eval.py
# This script processes CSV files in the srbench evaluation output directory

# Set the input directory containing CSV files
INPUT_DIR="cot_test/*.csv"
RESPONSE_COLUMN="response"
CORRECT_COLUMN="gold answer"
SPLIT_COLUMN="split"
OUTPUT="cleaned_results/accuracy_results.csv"

echo "Running clean_eval.py..."
echo "Input directory: $INPUT_DIR"
echo "Response column: $RESPONSE_COLUMN"
echo "Output file: $OUTPUT"

# Run the clean_eval.py script
python src/eval/acc.py --input_pattern "$INPUT_DIR" \
                          --response_column "$RESPONSE_COLUMN" \
                          --correct_column "$CORRECT_COLUMN" \
                          --split_column "$SPLIT_COLUMN" \
                          --output "$OUTPUT"

echo "Clean evaluation completed!"