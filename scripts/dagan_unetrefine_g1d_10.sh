#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jhc4318
#SBATCH --output=dagan-train%j.out

echo "Loading conda environment"
export PATH=/vol/bitbucket/${USER}/fyp/miniconda3/envs/dagan/bin/:$PATH
conda init bash
conda activate dagan
source /vol/cuda/8.0.61-cudnn.7.0.2/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Running script"
CUDA_VISIBLE_DEVICES=0 python -u train.py --model unet_refine --mask gaussian1d --maskperc 10

echo "Complete"