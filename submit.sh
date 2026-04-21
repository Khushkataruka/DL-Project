#!/bin/bash

#SBATCH --job-name=kh-gpu

#SBATCH --partition=gpu
#SBATCH --nodelist=node2
#SBATCH --error=/home/pkc/ML_Challenge_2025/STT_logs/test_gpu_error_train_%j.log
#SBATCH --output=/home/pkc/ML_Challenge_2025/STT_logs/test_gpu_job_output_train_%j.log
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=16

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024.2  
module load cuda-12.8

# ── Virtual Environment Setup (one-time: uncomment the venv creation lines) ──
# python -m venv ~/envs/stt_env
# source ~/envs/stt_env/bin/activate 
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install --no-input transformers datasets evaluate peft accelerate librosa soundfile bitsandbytes

# ── Activate existing venv ──
source ~/envs/stt_env/bin/activate

# ── Ensure PyTorch's bundled CUDA libs take priority over system libs ──
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH

# ── HuggingFace authentication (required for gated dataset) ──
# Load token securely from .env file
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found! HF_TOKEN not loaded natively."
fi

# ── Install any missing packages (--no-input avoids interactive prompts) ──
pip install -r requirements.txt --no-input

export TF_ENABLE_ONEDNN_OPTS=0
python train.py
