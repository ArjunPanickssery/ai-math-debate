#!/bin/bash
#SBATCH --job-name="job"
#SBATCH --output="transformer.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bbry-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 12:00:00

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"

module load python/3.11.4

#source /u/apanickssery/venv/bin/activate

export TORCH_SHOW_CPP_STACKTRACE=1

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16
# srun -N 1 -n 4 ./a.out > myjob.out
# py-torch example, --ntasks-per-node=1 --cpus-per-task=64
source /u/apanickssery/venv/bin/activate

# Get wandb key from the environment variable and log in with it
wandb
wandb login $WANDB_API_KEY

# export PYTHONPATH=$PYTHONPATH:/u/apanickssery/dictionary_learning/transformer

pip3 list

CUDA_VISIBLE_DEVICES=0,1,2,3 srun python3 models.py
# CUDA_VISIBLE_DEVICES=1 srun python3 transformer/one_layer_transformer_train.py --gpu 1 &
# CUDA_VISIBLE_DEVICES=2 srun python3 transformer/one_layer_transformer_train.py --gpu 2 &
# CUDA_VISIBLE_DEVICES=3 srun python3 transformer/one_layer_transformer_train.py --gpu 3 &
