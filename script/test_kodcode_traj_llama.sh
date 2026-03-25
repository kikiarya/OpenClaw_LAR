CUDA_VISIBLE_DEVICES=0 python evaluate_kodcode.py \
    --checkpoint weight/kodcode_traj_llama/merge_model \
    --mapping dataset/kodcode_traj_llama/token_mapping.json \
    --output_dir eval_output/kodcode_traj_llama \
    --model_name NousResearch/Meta-Llama-3.1-8B-Instruct \
    --samples 200 \
    --baseline \
    --equal_length


CUDA_VISIBLE_DEVICES=1 python evaluate_mbpp.py \
    --checkpoint weight/kodcode_traj_llama/merge_model \
    --mapping dataset/kodcode_traj_llama/token_mapping.json \
    --output_dir eval_output/kodcode_traj_llama \
    --model_name NousResearch/Meta-Llama-3.1-8B-Instruct \
    --sample 500 \
    --baseline \
    --gpu_memory_utilization 0.7
