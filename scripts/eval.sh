HF_HOME="/home/s2750560/RDS/huggingface/cache"
DATASET="stogian/srbench2"


models=(
	"HuggingFaceTB/SmolVLM-500M-Instruct"
	"HuggingFaceTB/SmolVLM-Instruct"
	"Qwen/Qwen2.5-VL-3B-Instruct"
	"Qwen/Qwen2.5-VL-7B-Instruct"
	"llava-hf/llava-1.5-7b-hf"
	"llava-hf/llava-v1.6-mistral-7b-hf"
	"Salesforce/instructblip-vicuna-7b"
	"HuggingFaceM4/Idefics3-8B-Llama3"
	"Salesforce/instructblip-vicuna-13b"
	"meta-llama/Llama-3.2-11B-Vision-Instruct"
)

for MODEL in "${models[@]}"; do
	echo "Running inference for model: $MODEL"
	python src/inference.py --model $MODEL --dataset $DATASET --batch_size 4
done

python src/eval_mini.py --dataset $DATASET

python src/eval_intern.py --model_name "OpenGVLab/InternVL2_5-8B-MPO" --dataset_name $DATASET --batch_size 4
python src/eval_intern.py --model_name "OpenGVLab/InternVL2_5-26B-MPO" --dataset_name $DATASET --batch_size 4
