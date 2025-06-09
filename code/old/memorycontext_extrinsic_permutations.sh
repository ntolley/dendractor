#!/bin/bash
#SBATCH -J extrinsic_permutations
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 50:00:00
#SBATCH --mem=300G
#SBATCH --account carney-sjones-condo
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-3

# Specify an output file
#SBATCH -o /users/ntolley/data/ntolley/dendractor/extrinsic_permutations/job_out/extrinsic_permutations-%j.out
#SBATCH -e /users/ntolley/data/ntolley/dendractor/extrinsic_permutations/job_out/extrinsic_permutations-%j.out

module load anaconda
module load cudnn
source ~/.bashrc
conda activate jaxley2

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
python memorycontext_extrinsic_permutations.py $SLURM_ARRAY_TASK_ID

export OUTPATH="/users/ntolley/data/ntolley/dendractor/extrinsic_permutations/job_out"
scontrol show job $SLURM_JOB_ID >> ${OUTPATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed >> ${OUTPATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.stats
