#!/bin/bash
#SBATCH -J memory_permutations
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 50:00:00
#SBATCH --mem=300G
#SBATCH --account carney-sjones-condo
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-7

# Specify an output file
#SBATCH -o /users/ntolley/data/ntolley/dendractor/memory_permutations/job_out/memory_permutations-%j.out
#SBATCH -e /users/ntolley/data/ntolley/dendractor/memory_permutations/job_out/memory_permutations-%j.out

module load anaconda
module load cudnn
source ~/.bashrc
conda activate jaxley2

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
python memory_permutations.py $SLURM_ARRAY_TASK_ID

export OUTPATH="/users/ntolley/data/ntolley/dendractor/memory_permutations_long/job_out"
scontrol show job $SLURM_JOB_ID >> ${OUTPATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${OUTPATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
