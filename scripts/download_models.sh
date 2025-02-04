#!/bin/bash
# This script downloads several Hugging Face models into separate folders
# under ../bin/models/llms/. Each model's folder is named after its identifier with '/' replaced by '_'.

# Check if huggingface-cli is available
if ! command -v huggingface-cli &>/dev/null; then
    echo "Error: huggingface-cli is not installed. Please install it with:"
    echo "       pip install huggingface_hub"
    exit 1
fi

# Base directory to download models into
download_base="../bin/models/llms"

# Create the base directory if it doesn't exist
mkdir -p "$download_base"

# List of model identifiers to download
models=(
    # "google/gemma-2-2b-it"
    "google/gemma-2-27b-it"
    # "mistralai/Mistral-Small-24B-Instruct-2501"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # "Qwen/Qwen2.5-14B-Instruct-1M"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
	# "stabilityai/stable-diffusion-3.5-large"
	# "stabilityai/stable-diffusion-3.5-large-turbo"
    # "black-forest-labs/FLUX.1-dev"
    # "dataautogpt3/OpenDalleV1.1"
)

# Loop through each model in the list
for model in "${models[@]}"; do
    # Create a local folder name by replacing "/" with "_"
    folder=$(echo "$model" | tr '/' '_')
    target_dir="$download_base/$folder"
    echo "Downloading model: $model"
    echo "Target folder: $target_dir"

    # Download the model using huggingface-cli
    huggingface-cli download "$model" --local-dir "$target_dir"

    echo "------------------------------------------"
done

echo "All models have been downloaded to $download_base."
