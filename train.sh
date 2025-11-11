#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --partition=gpu_h100
#SBATCH --time=0:10:00
#SBATCH --mem=84G
#SBATCH --exclusive
#SBATCH --job-name=andes
#SBATCH -o ./log/andes_%j.out

# =============== 生成时间戳 ===============
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_DIR="./log"
OUTPUT_DIR="./check_qwen3/checkpoints_${TIMESTAMP}"



# =============== 加载 Snellius 2023 工具链 + CUDA ===============
module load 2023
module load CUDA/12.4.0

nvidia-smi


source /home/khe/miniconda3/bin/activate qwen

torchrun --nproc_per_node=4 train.py \
  --dataset DGM4 \
  --epochs 10 \
  --batch-size 32 \
  --lr 5e-5 \
  --lr-scheduler cosine \
  --warmup-ratio 0.1 \
  --lr-scale 1.5 \
  --regular-weight 0.05 \
