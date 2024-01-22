#!/bin/bash

#SBATCH --account=project_2008494

#SBATCH --partition=gpu

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --mem=64G

#SBATCH --time=4:00:00

#SBATCH --gres=gpu:v100:1



module load tensorflow/2.13
module load tf-models-official/2.13
module load tensorflow_hub
module load transformers
srun python3 koulutus.py

