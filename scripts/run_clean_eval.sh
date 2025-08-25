#!/bin/bash

# Simple bash script to run clean_eval.py
# This script processes CSV files in the srbench evaluation output directory

# Set the input directory containing CSV files
INPUT_DIR="output/evaluations/srbench/"

# Set the response column name (adjust if needed)
RESPONSE_COLUMN="response"

echo "Running clean_eval.py..."
echo "Input directory: $INPUT_DIR"
echo "Response column: $RESPONSE_COLUMN"

# Run the clean_eval.py script
python src/eval/llmaj.py \
    --input_directory "$INPUT_DIR" \
    --response-column "$RESPONSE_COLUMN"\
    --model "meta-llama/Llama-3.1-8B-Instruct"

echo "Clean evaluation completed!"