python identify_segments.py \
    --input_jsonl dataset/mind2web_qwen/qwen_traj_original.jsonl \
    --output dataset/mind2web_qwen/segments.json \
    --include_conversations \
    --n_min 2 \
    --n_max 6 \
    --max_segments 100 \
    --min_freq 1000

python create_token_mapping.py \
    --segments dataset/mind2web_qwen/segments.json \
    --output dataset/mind2web_qwen/token_mapping.json 

python build_sft_dataset_v1.py \
    --input_jsonl dataset/mind2web_qwen/qwen_traj_original.jsonl \
    --mapping dataset/mind2web_qwen/token_mapping.json \
    --output_dir dataset/mind2web_qwen \
    --compress_system \
    --compress_conversations \
    --split_ratio 0.99

CUDA_VISIBLE_DEVICES=3 \
python train.py \
    --dataset_dir dataset/triviaqa_llama \
    --mapping dataset/triviaqa_llama/token_mapping.json \
    --output_dir weight/triviaqa_llama/full \
    --train_new_embeddings_only \
    --model_name NousResearch/Meta-llama-3.1-8B-Instruct \
    --teacher_model_name NousResearch/Meta-llama-3.1-8B-Instruct \
    --epochs 3 \
    --use_kl_distillation \
    --batch_size 4 \
    --kl_weight 1.0


    --model_name NousResearch/Meta-llama-3.1-8B-Instruct \
    --teacher_model_name NousResearch/Meta-llama-3.1-8B-Instruct \


python build_sft_dataset_v1.py \
    --input_jsonl dataset/triviaqa_llama/original_mind2web_sample.jsonl \
    --mapping dataset/triviaqa_llama/token_mapping.json \
    --output_dir dataset/triviaqa_llama \
    --compress_system \
    --compress_conversations \
    --split_ratio 0.99

CUDA_VISIBLE_DEVICES=0 \
python merge_lora_embedding.py \
    --base_model Qwen/Qwen3-8B \
    --lora_checkpoint weight/mind2web_qwen/full/final_checkpoint \
    --output_dir weight/mind2web_qwen/merged_model \
    --token_mapping dataset/mind2web_qwen/token_mapping.json

python identify_segments.py     --input_jsonl dataset/triviaqa_traj_llama3_20/original_llama3_sample.jsonl     --output dataset/triviaqa_traj_llama3_20/segments.json     --include_conversations     --n_min 3     --n_max 5  --no_html   --max_segments 1000 --min_freq 2000

python create_token_mapping.py     --segments dataset/triviaqa_traj_llama3_20/segments.json     --output dataset/triviaqa_traj_llama3_20/token_mapping.json

python build_sft_dataset_v1.py     --input_jsonl dataset/triviaqa_traj_llama3_20/original_llama3_sample.jsonl     --mapping dataset/triviaqa_traj_llama3_20/token_mapping.json     --output_dir dataset/triviaqa_traj_llama3_20     --compress_system     --compress_conversations     --split_ratio 0.99

CUDA_VISIBLE_DEVICES=2 python train.py     --dataset_dir dataset/triviaqa_traj_llama3_20     --mapping dataset/triviaqa_traj_llama3_20/token_mapping.json     --output_dir weight/triviaqa_traj_llama3_20/full     --train_new_embeddings_only     --model_name llama/llama3-8B     --teacher_model_name llama/llama3-8B     --epochs 3     --use_kl_distillation     --batch_size 2     --kl_weight 1.0

python build_sft_dataset_v1.py     --input_jsonl dataset/triviaqa_traj_llama3_1000/original_llama8b_sample.jsonl     --mapping dataset/triviaqa_traj_llama3_1000/token_mapping.json     --output_dir dataset/triviaqa_traj_llama3_1000/     --compress_system     --compress_conversations     --split_ratio 0.99

CUDA_VISIBLE_DEVICES=4 python train.py     --dataset_dir dataset/triviaqa_traj_llama3_1000     --mapping dataset/triviaqa_traj_llama3_1000/token_mapping.json     --output_dir weight/triviaqa_traj_llama3_1000/full     --train_new_embeddings_only     --model_name llama/llama3-8B     --teacher_model_name llama/llama3-8B     --epochs 1     --use_kl_distillation     --batch_size 1     --kl_weight 1.0

CUDA_VISIBLE_DEVICES=0 \
python merge_lora_embedding.py \
    --base_model llama/llama3-8B \
    --lora_checkpoint weight/triviaqa_traj_llama3_1000/full/final_checkpoint \
    --output_dir weight/triviaqa_traj_llama3_1000/merged_model \
    --token_mapping dataset/triviaqa_traj_llama3_1000/token_mapping.json

CUDA_VISIBLE_DEVICES=1 \
python evaluate_mind2web.py \
    -checkpoint weight/mind2web_qwen/merge_model \
    --model_name Qwen/Qwen3-8B \
    --mapping dataset/mind2web_qwen/token_mapping.json \
    --output_dir eval_output/mind2web_qwen \
    --dataset  \
    --samples 1000 \
    --gpu_memory_utilization 0.8 \
    --equal_length

CUDA_VISIBLE_DEVICES=2 python evaluate_mind2web.py     --checkpoint weight/mind2web_qwen/merge_model     --mapping dataset/mind2web_qwen/token_mapping.json     --output_dir eval_output/mind2web_qwen     --model_name Qwen/Qwen3-8B     --sample 10000     --gpu_memory_utilization 0.8     --test_split all