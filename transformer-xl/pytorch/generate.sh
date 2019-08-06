#!/bin/bash

#SBATCH -A umutlu
#SBATCH --job-name=generate_text
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=short
#SBATCH --output=generate-%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
# #SBATCH --mem=2500
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=urasmutlu@gmail.com

# module load centos7.3/lib/cuda/9.0

srun python inference.py