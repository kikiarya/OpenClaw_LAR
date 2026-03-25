python evaluate_mind2web.py \
    --checkpoint weight/mind2web_traj_llama/merge_model \
    --mapping dataset/mind2web_traj_llama/token_mapping.json \
    --output_dir eval_output/mind2web_traj_llama \
    --model_name NousResearch/Meta-Llama-3.1-8B-Instruct \
    --sample 456 \
    --baseline \
    --gpu_memory_utilization 0.5


CUDA_VISIBLE_DEVICES=0 python evaluate_mind2web.py \
    --checkpoint weight/mind2web_traj_llama_nothink/merge_model \
    --mapping dataset/mind2web_traj_llama_nothink/token_mapping.json \
    --output_dir eval_output/mind2web_traj_llama_nothink \
    --model_name NousResearch/Meta-Llama-3.1-8B-Instruct \
    --sample 50 \
    --gpu_memory_utilization 0.8 \
    --baseline \
    --test_split domain