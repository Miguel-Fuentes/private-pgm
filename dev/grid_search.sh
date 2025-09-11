#!/bin/bash
#SBATCH --job-name=pgm_grid
#SBATCH --output=grid_results_%A_%a.out
#SBATCH --error=grid_results_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=04:00:00         # 4 hours max time
#SBATCH --mem=4G                # 4GB memory per job
#SBATCH --cpus-per-task=2       # 2 CPUs per job
# Add any other SLURM options you need (partition, etc.)

# Define parameter values
MANDATORY_ARGS=(0.01 0.1 1.0 10.0 100.0)
MAX_ITERS=(500 1000 5000 10000)

# Calculate which parameters to use based on array index
MANDATORY_IDX=$((SLURM_ARRAY_TASK_ID / 4))
MAX_ITER_IDX=$((SLURM_ARRAY_TASK_ID % 4))

MANDATORY=${MANDATORY_ARGS[$MANDATORY_IDX]}
MAX_ITER=${MAX_ITERS[$MAX_ITER_IDX]}

echo "Running with mandatory_arg=$MANDATORY, max_iters=$MAX_ITER"

# Activate virtual environment and run the script
source ~/private-pgm/.venv/bin/activate
python ~/private-pgm/dev/run.py $MANDATORY --max_iters=$MAX_ITER

echo "Job completed"
