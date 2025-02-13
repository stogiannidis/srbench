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
download_base="../bin/models/vlms"

# Create the base directory if it doesn't exist
mkdir -p "$download_base"

# List of model identifiers to download
models=(
    # "google/gemma-2-2b-it"
    # "google/gemma-2-27b-it"
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
    # deepseek-ai/Janus-Pro-7B
    # allenai/Llama-3.1-Tulu-3-8B
    # allenai/OLMo-2-1124-13B-Instruct
    # DeepFloyd/IF-II-L-v1.0
    # Qwen/Qwen2.5-VL-3B-Instruct
    # Qwen/Qwen2.5-VL-7B-Instruct
    # openbmb/MiniCPM-V-2_6
    # HuggingFaceM4/Idefics3-8B-Llama3
    # OpenGVLab/InternVL2_5-8B
    # microsoft/Phi-3.5-vision-instruct
    # deepseek-ai/deepseek-vl2-tiny
    # deepseek-ai/deepseek-vl2
    # THUDM/cogvlm2-llama3-chat-19B
)

# Loop through each model in the list
for model in "${models[@]}"; do
    # Create a local folder name by removing everything before '/' and keeping only the part after it
    folder="${model#*/}"
    target_dir="$download_base/$folder"
    echo "Downloading model: $model"
    echo "Target folder: $target_dir"

    # Download the model using huggingface-cli
    huggingface-cli download "$model" --local-dir "$target_dir"

    echo "------------------------------------------"
done

echo "All models have been downloaded to $download_base."