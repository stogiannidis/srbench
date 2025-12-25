#!/bin/bash

# Configuration
DATASET="stogian/srbenchv2"
CONCURRENCY=100
SCRIPT_PATH="src/infer.py"

# Define the models to run
MODELS=(
	"nvidia/nemotron-nano-12b-v2-vl:free"
	"x-ai/grok-4.1-fast"
	"openai/gpt-5.2"
    "anthropic/claude-sonnet-4.5"
	"google/gemini-3-flash-preview"
    "mistralai/mistral-small-3.1-24b-instruct:free"
)

# Optional: Set to "true" to enable chain-of-thought prompting
USE_COT=true

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting inference for ${#MODELS[@]} models${NC}"
echo -e "${BLUE}Dataset: ${DATASET}${NC}"
echo -e "${BLUE}Batch Size: ${CONCURRENCY}${NC}"
echo -e "${BLUE}Chain-of-Thought: ${USE_COT}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Track successful and failed runs
SUCCESS_COUNT=0
FAILED_MODELS=()

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo -e "${GREEN}Processing model: ${MODEL}${NC}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Build the command
    CMD="python ${SCRIPT_PATH} --dataset ${DATASET} --model ${MODEL} --max_concurrency ${CONCURRENCY}"
    
    # Add --cot flag if enabled
    if [ "$USE_COT" = true ]; then
        CMD="${CMD} --cot"
    fi
    
    # Run the inference
    if eval $CMD; then
        echo -e "${GREEN}✓ Successfully completed ${MODEL}${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ Failed to process ${MODEL}${NC}"
        FAILED_MODELS+=("${MODEL}")
    fi
    
    echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BLUE}----------------------------------------${NC}\n"
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Inference Complete${NC}"
echo -e "${GREEN}Successful runs: ${SUCCESS_COUNT}/${#MODELS[@]}${NC}"

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo -e "${RED}Failed models:${NC}"
    for FAILED_MODEL in "${FAILED_MODELS[@]}"; do
        echo -e "${RED}  - ${FAILED_MODEL}${NC}"
    done
fi

echo -e "${BLUE}========================================${NC}"