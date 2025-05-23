#!/bin/bash

# Optimized run script with error handling and parallel processing
set -euo pipefail

# Configuration
DATASET="stogian/srbench2"
BATCH_SIZE=4
MAX_WORKERS=2
LOG_DIR="logs"
RESULTS_DIR="output/evaluations"

# Create necessary directories
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/run.log"
}

# Function to check GPU memory
check_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        local memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        local memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        local memory_percent=$((memory_usage * 100 / memory_total))
        
        if [ $memory_percent -gt 90 ]; then
            log "WARNING: GPU memory usage is high ($memory_percent%)"
            return 1
        fi
    fi
    return 0
}

# Function to run inference with error handling
run_inference() {
    local model="$1"
    local model_name=$(basename "$model")
    local log_file="$LOG_DIR/${model_name}.log"
    
    log "Starting inference for model: $model"
    
    # Check if results already exist
    if [ -f "$RESULTS_DIR/$(basename $DATASET)/${model_name}.csv" ]; then
        log "Results already exist for $model_name, skipping..."
        return 0
    fi
    
    # Check GPU memory before starting
    if ! check_gpu_memory; then
        log "GPU memory too high, waiting..."
        sleep 30
    fi
    
    # Run inference with timeout and error handling
    if timeout 3600 python src/inference.py \
        --model "$model" \
        --dataset "$DATASET" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$MAX_WORKERS" \
        2>&1 | tee "$log_file"; then
        log "Successfully completed inference for $model"
    else
        local exit_code=$?
        log "ERROR: Inference failed for $model (exit code: $exit_code)"
        return $exit_code
    fi
}

# Optimized model list with memory-efficient ordering
models=(
    "HuggingFaceTB/SmolVLM-500M-Instruct"
    "HuggingFaceTB/SmolVLM-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "llava-hf/llava-1.5-7b-hf"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "Salesforce/instructblip-vicuna-7b"
    "HuggingFaceM4/Idefics3-8B-Llama3"
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "Salesforce/instructblip-vicuna-13b"
)

# Function to run models in parallel (with limited concurrency)
run_models_parallel() {
    local max_parallel=1  # Adjust based on GPU memory
    local pids=()
    
    for model in "${models[@]}"; do
        # Wait if we've reached max parallel processes
        while [ ${#pids[@]} -ge $max_parallel ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[i]}" 2>/dev/null; then
                    unset "pids[i]"
                fi
            done
            pids=("${pids[@]}")  # Reindex array
            sleep 5
        done
        
        # Start new process
        run_inference "$model" &
        pids+=($!)
        
        # Small delay to prevent resource conflicts
        sleep 10
    done
    
    # Wait for all remaining processes
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

# Main execution
main() {
    log "Starting optimized evaluation run"
    log "Dataset: $DATASET"
    log "Batch size: $BATCH_SIZE"
    log "Max workers: $MAX_WORKERS"
    
    # Check dependencies
    if ! python -c "import torch; print(f'PyTorch {torch.__version__} available')" 2>/dev/null; then
        log "ERROR: PyTorch not available"
        exit 1
    fi
    
    # Run standard models
    run_models_parallel
    
    # Run specialized evaluations
    log "Running MiniCPM evaluation"
    python src/eval_mini.py --dataset "$DATASET" 2>&1 | tee "$LOG_DIR/minicpm.log"
    
    log "Running InternVL evaluations"
    for model in "OpenGVLab/InternVL2_5-8B-MPO" "OpenGVLab/InternVL2_5-26B-MPO"; do
        model_name=$(basename "$model")
        log "Running InternVL evaluation for $model"
        python src/eval_intern.py \
            --model_name "$model" \
            --dataset_name "$DATASET" \
            --batch_size "$BATCH_SIZE" \
            2>&1 | tee "$LOG_DIR/intern_${model_name}.log"
    done
    
    log "All evaluations completed successfully"
}

# Trap signals for cleanup
trap 'log "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"
