DATASET="stogian/srbenchv2"
ONESHOT_PATH="/netdisk/users/stogian/srbench/example/oneshot.json"

models=(
	"MLL-Lab/MindCube-Qwen2.5VL-RawQA-SFT"
	# "OpenGVLab/InternVL3_5-8B-HF"
	# "OpenGVLab/InternVL3_5-14B-HF"
    # "OpenGVLab/InternVL3_5-38B-HF"
    # "OpenGVLab/InternVL3_5-30B-A3B-HF"
    # "OpenGVLab/InternVL3_5-241B-A28B-HF"
    # "google/gemma-3-4b-it"
    # "google/gemma-3-12b-it"
    # "google/gemma-3-27b-it"
    # "Qwen/Qwen3-VL-8B-Thinking"
    # "Qwen/Qwen3-VL-30B-A3B-Thinking"
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # "Qwen/Qwen3-VL-235B-A22B-Thinking"
    # "Qwen/Qwen3-VL-235B-A22B-Instruct"
	# "HuggingFaceM4/Idefics3-8B-Llama3"
	# "llava-hf/llava-1.5-7b-hf"
	# "llava-hf/llava-v1.6-mistral-7b-hf"
	# "llava-hf/llava-onevision-qwen2-7b-ov-hf"
	# "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
	# "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
	# "meta-llama/Llama-3.2-11B-Vision-Instruct"
	# "meta-llama/Llama-3.2-90B-Vision-Instruct"
	# "openbmb/MiniCPM-V-4_5"
	# "zai-org/GLM-4.6V-Flash"
	# "moonshotai/Kimi-VL-A3B-Instruct"
)

BATCH_SIZE=32

for MODEL in "${models[@]}"; do
	echo "Running inference for model: $MODEL"


	# python src/eval.py --model $MODEL \
	# 					--dataset $DATASET \
	# 					--batch_size $BATCH_SIZE \
	# 					--seed 123 \
    #                     --num_workers 10 

	python src/eval.py --model $MODEL \
						--dataset $DATASET \
						--cot \
						--structured-output json \
						--batch_size $BATCH_SIZE \
						--seed 123 \
                        --num_workers 18 \
						--max_samples 100
						# --one-shot $ONESHOT_PATH \




done

