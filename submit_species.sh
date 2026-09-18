#!/bin/bash
# submit_species.sh
# SLURM job script for species_tracker.py
# Adjust partition, time, and paths for your cluster

#SBATCH --job-name=species_track
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/species_%j.out
#SBATCH --error=logs/species_%j.err

module load python/3.11      # adjust to your cluster's module name

# Activate your venv / conda env if applicable
# source ~/envs/analysis/bin/activate

TRAJ=/path/to/your/traj.dump
OUTDIR=/path/to/output/dir

mkdir -p "$OUTDIR" logs

python species_tracker.py \
    --dump    "$TRAJ" \
    --output  "$OUTDIR/species.csv" \
    --cutoff  2.2 \
    --stride  100 \
    --zmax    60.0 \
    --plot

# -----------------------------------------------------------------------
# Notes:
# --stride 100 means analyse 1 in every 100 frames.
# For a 4M step run with dump every 1000 steps = 4000 frames total,
# stride 10 gives 400 analysis points — usually plenty.
# --zmax sets a z ceiling to exclude gas-phase molecules far from surface.
# -----------------------------------------------------------------------
