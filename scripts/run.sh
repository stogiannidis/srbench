DATASET="stogian/srbench2"
ONESHOT_PATH="/netdisk/users/stogian/srbench/example/oneshot.json"

models=(
	"OpenGVLab/InternVL3_5-8B-HF"
	"OpenGVLab/InternVL3_5-14B-HF"
    "OpenGVLab/InternVL3_5-38B-HF"
    "OpenGVLab/InternVL3_5-30B-A3B-HF"
    "OpenGVLab/InternVL3_5-241B-A28B-HF"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "Qwen/Qwen3-VL-8B-Thinking"
    "Qwen/Qwen3-VL-30B-A3B-Thinking"
    "Qwen/Qwen3-VL-235B-A22B-Thinking"
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
    "Qwen/Qwen3-VL-235B-A22B-Instruct"
	"meta-llama/Llama-3.2-11B-Vision-Instruct"
	"meta-llama/Llama-3.2-90B-Vision-Instruct"
	"HuggingFaceM4/Idefics3-8B-Llama3"
	"llava-hf/llava-1.5-7b-hf"
	"llava-hf/llava-v1.6-mistral-7b-hf"
	"openbmb/MiniCPM-V-2_6"
	"HuggingFaceTB/SmolVLM-500M-Instruct"
	"HuggingFaceTB/SmolVLM-Instruct"
	"moonshotai/Kimi-VL-A3B-Thinking-2506"
	"moonshotai/Kimi-VL-A3B-Instruct"
	"zai-org/GLM-4.1V-9B-Thinking"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    "meta-llama/Llama-4-Scout-17B-128E-Instruct"
)

for MODEL in "${models[@]}"; do
	echo "Running inference for model: $MODEL"

	    # Determine optimal batch size based on model size
    if [[ $MODEL == *"11B"* ]]; then
        BATCH_SIZE=16
    elif [[ $MODEL == *"3B"* ]]; then
        BATCH_SIZE=32
    elif [[ $MODEL == *"8B"* ]]; then
        BATCH_SIZE=16
    else
        BATCH_SIZE=8
    fi

	python src/eval.py --model $MODEL \
						--dataset $DATASET \
						--batch_size $BATCH_SIZE \
						--seed 123 \
                        --num_workers 8 

	python src/eval.py --model $MODEL \
						--dataset $DATASET \
						--cot \
						--batch_size $BATCH_SIZE \
						--seed 123 \
                        --num_workers 8 




done

