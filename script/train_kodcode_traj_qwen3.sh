python train.py \
    --dataset_dir dataset/kodcode_traj_qwen3 \
    --mapping dataset/kodcode_traj_qwen3/token_mapping.json \
    --output_dir weight/kodcode_traj_qwen3/full \
    --train_new_embeddings_only \
    --epochs 1 \
    --use_kl_distillation \
    --kl_weight 1.0 \
    --batch_size 1