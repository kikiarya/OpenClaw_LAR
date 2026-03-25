python train.py \
    --dataset_dir dataset/triviaqa_traj_qwen3 \
    --mapping dataset/triviaqa_traj_qwen3/token_mapping.json \
    --output_dir weight/triviaqa_traj_qwen3/full \
    --train_new_embeddings_only \
    --epochs 1 \
    --use_kl_distillation \
    --kl_weight 1.0 \
    --batch_size 4