DATASET="stogian/srbench"
ONESHOT_PATH="/netdisk/users/stogian/srbench/example/oneshot.json"

models=(
	"OpenGVLab/InternVL2_5-26B-MPO"
	"meta-llama/Llama-3.2-11B-Vision-Instruct"
	"Qwen/Qwen2.5-VL-3B-Instruct"
	"Qwen/Qwen2.5-VL-7B-Instruct"
	"HuggingFaceM4/Idefics3-8B-Llama3"
	"llava-hf/llava-1.5-7b-hf"
	"llava-hf/llava-v1.6-mistral-7b-hf"
	"OpenGVLab/InternVL2_5-8B-MPO"
	"openbmb/MiniCPM-V-2_6"
	"HuggingFaceTB/SmolVLM-500M-Instruct"
	"HuggingFaceTB/SmolVLM-Instruct"
	"Salesforce/instructblip-vicuna-7b"
	"Salesforce/instructblip-vicuna-13b"
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


	# python src/eval.py --model $MODEL \
	# 						--dataset $DATASET \
	# 						--one-shot $ONESHOT_PATH \
	# 						--cot \
	# 						--batch_size $BATCH_SIZE

	python src/eval.py --model $MODEL \
						--dataset $DATASET \
						--batch_size $BATCH_SIZE \
						--seed 123 \
						--max_samples 10
done
