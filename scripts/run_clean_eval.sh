#!/usr/bin/env bash
# Wrapper script to run the MultipleChoiceNormalizer cleaning over all CSV files in a directory.
# The underlying Python script already iterates over every *.csv in the provided directory,
# so we only need to call it once.
#
# Usage:
#   scripts/run_clean_eval.sh -d <directory> [-r <response_column>] [-c "A B C D E"] [--no-few-shot] [--max-samples N]
# Examples:
#   scripts/run_clean_eval.sh -d output/evaluations/srbench
#   scripts/run_clean_eval.sh -d output/evaluations/srbench -r raw_response -c "Yes No"
#
# Notes:
#  - Choices should be provided as a quoted, space-separated string.
#  - Pass --no-few-shot to disable few-shot examples.
#  - --max-samples is useful for quick tests.

set -euo pipefail

DIR=""
RESP_COL="raw_response"
CHOICES="A B C D E"
NO_FEW_SHOT=""
MAX_SAMPLES=""
PYTHON_BIN="python"

print_help() {
  grep '^# ' "$0" | sed 's/^# //'
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--directory)
      DIR="$2"; shift 2;;
    -r|--response-column)
      RESP_COL="$2"; shift 2;;
    -c|--choices)
      CHOICES="$2"; shift 2;;
    --no-few-shot)
      NO_FEW_SHOT="--no-few-shot"; shift 1;;
    --max-samples)
      MAX_SAMPLES="--max-samples $2"; shift 2;;
    -p|--python)
      PYTHON_BIN="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; print_help; exit 1;;
  esac
done

if [[ -z "$DIR" ]]; then
  echo "Error: directory not specified" >&2
  print_help
  exit 1
fi

if [[ ! -d "$DIR" ]]; then
  echo "Error: directory does not exist: $DIR" >&2
  exit 1
fi

# Convert choices to array for safe passing
read -r -a CHOICE_ARR <<< "$CHOICES"

echo "Running clean_eval on directory: $DIR"
set -x
$PYTHON_BIN src/eval/clean_eval.py \
  --input_directory "$DIR" \
  --response-column "$RESP_COL" \
  --choices "${CHOICE_ARR[@]}" \
  $NO_FEW_SHOT \
  $MAX_SAMPLES
set +x

echo "Done."