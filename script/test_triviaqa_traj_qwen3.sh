CUDA_VISIBLE_DEVICES=3 \
python evaluate_triviaqa.py \
    --checkpoint weight/triviaqa_qwen/merge_model \
    --model_name Qwen/Qwen3-8B \
    --mapping dataset/triviaqa_qwen/token_mapping.json \
    --output_dir eval_output/musique_qwen \
    --dataset musique \
    --samples 1000 \
    --gpu_memory_utilization 0.8 \
    --baseline
    # --max_new_tokens 8192 \
