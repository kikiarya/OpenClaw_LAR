CUDA_VISIBLE_DEVICES=1 python evaluate_kodcode.py \
    --checkpoint weight/kodcode_traj_qwen3/merge_model \
    --mapping dataset/kodcode_traj_qwen3/token_mapping.json \
    --output_dir eval_output/kodcode_traj_qwen3 \
    --samples 200 \
    --equal_length
