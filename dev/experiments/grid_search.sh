#!/bin/bash
#SBATCH --job-name=pgm_grid_robust
#SBATCH --output=grid_results_%A_%a.out
#SBATCH --error=grid_results_%A_%a.err
# Note: The array range is now calculated automatically below
#SBATCH --time=04:00:00       # 4 hours max time
#SBATCH --mem=4G              # 4GB memory per job
#SBATCH --cpus-per-task=2     # 2 CPUs per job

# --- Define parameter values ---
# You can change these arrays without touching the logic below
SIZE_MULTIPLIERS=(0.1 0.3981 1.5849 6.3096 25.1189 100.0)
LAMBDA_REGS=(1e-2 1e-4 1e-6 1e-8) # Added a 4th value to show it works

# --- Automatic Calculation Logic ---
# 1. Get the number of elements in each parameter array
NUM_MULTIPLIERS=${#SIZE_MULTIPLIERS[@]}
NUM_LAMBDAS=${#LAMBDA_REGS[@]}

# 2. Calculate the total number of jobs needed
#    SLURM arrays are 0-indexed, so we subtract 1
TOTAL_JOBS=$((NUM_MULTIPLIERS * NUM_LAMBDAS))
SLURM_ARRAY_MAX=$((TOTAL_JOBS - 1))

# 3. Dynamically set the job array size (for information/submission)
#    Note: You still need to set this in your sbatch command, e.g., `sbatch --array=0-$SLURM_ARRAY_MAX my_script.sh`
#    This line is mostly for logging and clarity.
echo "This script is configured for a job array of 0-$SLURM_ARRAY_MAX"

# --- Dynamic Indexing ---
# Calculate which parameters to use based on the SLURM array task ID
MULTIPLIER_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LAMBDAS))
LAMBDA_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LAMBDAS))

# Retrieve the actual parameter values from the arrays
SIZE_MULTIPLIER=${SIZE_MULTIPLIERS[$MULTIPLIER_IDX]}
LAMBDA_REG=${LAMBDA_REGS[$LAMBDA_IDX]}

echo "Running task $SLURM_ARRAY_TASK_ID with size_multiplier=$SIZE_MULTIPLIER, lambda_reg=$LAMBDA_REG"

# Activate your virtual environment and run the Python script
source ~/private-pgm/.venv/bin/activate
python ~/private-pgm/dev/run.py $SIZE_MULTIPLIER --lambda_reg=$LAMBDA_REG

echo "Job completed"