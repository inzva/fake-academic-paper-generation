#!/bin/bash

#SBATCH -A umutlu
#SBATCH --job-name=transformer-xl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=akya-cuda
#SBATCH --output=transformer-%j.out
#SBATCH --gres=gpu:4
#SBATCH --time=06-00:00:00
# #SBATCH --mem=2500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=urasmutlu@gmail.com

# module load centos7.3/lib/cuda/9.0

module load /truba/home/umutlu/cuda_9.0_module

srun bash run_papers_base.sh train --work_dir experiments