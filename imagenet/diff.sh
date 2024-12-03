#!/bin/bash
#SBATCH --job-name=test_my_program
#SBATCH --account=project_2012241
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80000
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=test_out.log
#SBATCH --error=test_err.log

module load pytorch/2.2
#source /scratch/project_2012241/paper/bin/activate

export PYTHONPATH=/scratch/project_2012241/paper/lib/python3.9/site-packages:$PYTHONPATH

srun python3 diff.py
