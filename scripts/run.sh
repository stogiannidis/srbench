DATASET="benchmark/srbench"
BATCH_SIZE=16

models=(
	"OpenGVLab/InternVL3-8B"
	"OpenGVLab/InternVL3-38B"
	"OpenGVLab/InternVL3-77B"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
	"Qwen/Qwen2.5-VL-7B-Instruct"
	"Qwen/Qwen2.5-VL-32B-Instruct"
	"Qwen/Qwen2.5-VL-72B-Instruct"
	"meta-llama/Llama-3.2-11B-Vision-Instruct"
	"meta-llama/Llama-3.2-90B-Vision-Instruct"
	"HuggingFaceM4/Idefics3-8B-Llama3"
	"llava-hf/llava-1.5-7b-hf"
	"llava-hf/llava-v1.6-mistral-7b-hf"
	"openbmb/MiniCPM-V-2_6"
	"HuggingFaceTB/SmolVLM-500M-Instruct"
	"HuggingFaceTB/SmolVLM-Instruct"
)

for MODEL in "${models[@]}"; do
	echo "Running inference for model: $MODEL"


	python src/eval.py --model $MODEL \
						--dataset $DATASET \
						--cot \
						--batch_size $BATCH_SIZE \
						--seed 123


done

