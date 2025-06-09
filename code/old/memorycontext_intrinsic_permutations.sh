#!/bin/bash
#SBATCH -J intrinsic_permutations
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 50:00:00
#SBATCH --mem=300G
#SBATCH --account carney-sjones-condo
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-8

# Specify an output file
#SBATCH -o /users/ntolley/data/ntolley/dendractor/intrinsic_permutations/job_out/intrinsic_permutations-%j.out
#SBATCH -e /users/ntolley/data/ntolley/dendractor/intrinsic_permutations/job_out/intrinsic_permutations-%j.out

module load anaconda
module load cudnn
source ~/.bashrc
conda activate jaxley2

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
python memorycontext_intrinsic_permutations.py $SLURM_ARRAY_TASK_ID

export OUTPATH="/users/ntolley/data/ntolley/dendractor/intrinsic_permutations/job_out"
scontrol show job $SLURM_JOB_ID >> ${OUTPATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${OUTPATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
