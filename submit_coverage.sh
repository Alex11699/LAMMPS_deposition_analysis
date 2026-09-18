#!/bin/bash
# submit_coverage.sh
# SLURM job script for surface_coverage.py

#SBATCH --job-name=surf_coverage
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/coverage_%j.out
#SBATCH --error=logs/coverage_%j.err

module load python/3.11

# source ~/envs/analysis/bin/activate

TRAJ=/path/to/your/traj.dump
OUTDIR=/path/to/output/dir

mkdir -p "$OUTDIR" logs

python surface_coverage.py \
    --dump             "$TRAJ" \
    --output           "$OUTDIR/coverage.csv" \
    --adsorbate-types  1 2 3 \
    --substrate-types  4 5 \
    --surface-z-ref    auto \
    --slab-thickness   3.0 \
    --band-width       8.0 \
    --stride           50 \
    --plot

# -----------------------------------------------------------------------
# Notes:
# --band-width: the z-slab above the surface counted as "on surface".
#   8 Å typically captures chemisorbed + weakly physisorbed species.
#   Increase to ~15 Å if you want to include the second monolayer.
# --surface-z-ref auto: recalculates the surface position each frame,
#   which is important if the slab drifts or expands during deposition.
#   Replace with a fixed value (e.g. --surface-z-ref 35.4) if the slab
#   is held rigid and you want a stable reference.
# -----------------------------------------------------------------------
