torchrun --nproc_per_node=2 train.py \
  --dataset DGM4 \
  --epochs 20 \
  --batch-size 2 \
  --lr 5e-5 \
  --lr-scheduler cosine \
  --warmup-ratio 0.1 \
  --lr-scale 1.5 \
  --regular-weight 0.05 \
