CUDA_VISIBLE_DEVICES=0 python evaluate_triviaqa.py \
    --checkpoint weight/triviaqa_traj_llama/merge_model \
    --mapping dataset/triviaqa_traj_llama/token_mapping.json \
    --output_dir eval_output/triviaqa_traj_llama \
    --model_name NousResearch/Meta-Llama-3.1-8B-Instruct \
    --sample 163 \
    --baseline \
    --gpu_memory_utilization 0.8
    # --max_new_tokens 8192 \
