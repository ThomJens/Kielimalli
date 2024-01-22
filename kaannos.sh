#!/bin/bash

#SBATCH --account=project_2008494

#SBATCH --partition=gpu

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --mem=64G

#SBATCH --time=4:00:00

#SBATCH --gres=gpu:v100:1



module load pytorch/2.0
module load sentencepiece transformers/4.26.1

srun python3 kaannos.py

