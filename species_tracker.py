"""
species_tracker.py
------------------
Track molecular species formation from LAMMPS dump files using OVITO.
Designed for TMA deposition onto SiO2: identifies fragments by
connectivity (cluster analysis) and composition (atom type counts).

Usage:
    python species_tracker.py --dump traj.dump --output species.csv
    python species_tracker.py --dump traj.dump --output species.csv \
        --cutoff 2.0 --stride 100 --zmax 30.0

Requires: ovito (free tier), numpy, pandas
Install:  pip install ovito numpy pandas
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

import ovito
from ovito.io import import_file
from ovito.modifiers import (
    ClusterAnalysisModifier,
    ExpressionSelectionModifier,
    DeleteSelectedModifier,
)
from ovito.data import DataCollection


# ---------------------------------------------------------------------------
# Configuration — edit these to match your LAMMPS atom types
# ---------------------------------------------------------------------------

# Map LAMMPS numeric type -> element symbol
# Adjust to match your force field ordering
ATOM_TYPE_MAP = {
    1: "H",    # substrate hydroxyl H
    2: "O",    # substrate oxygen
    3: "Si",   # substrate silicon
    4: "Al",   # TMA aluminium
    5: "C",    # methyl carbon
    6: "H",    # methyl hydrogen (TMA-side)
    # Note: types 1 and 6 both map to "H" — they collapse to the same
    # element in composition fingerprints, which is correct for species
    # labelling. SUBSTRATE_TYPES below keeps them separate for
    # surface-bound vs gas-phase classification.
}

# Bonding cutoffs (Angstrom) for cluster analysis
# Used as a single global cutoff; ovito ClusterAnalysis uses one value.
# Set to capture the longest bond in your system (typically Al-C ~2.0 Å).
DEFAULT_CUTOFF = 2.2

# Substrate atom types — atoms of these types are excluded from the
# adsorbate composition fingerprint and used to detect surface binding.
SUBSTRATE_TYPES = {1, 2, 3}  # substrate H, O, Si

# Known TMA-derived species fingerprints: frozenset of (element, count) tuples
# Compositions use adsorbate atoms only (types 4, 5, 6 → Al, C, H).
KNOWN_SPECIES = {
    frozenset([("Al", 1), ("C", 3), ("H", 9)]):  "TMA",           # Al(CH3)3
    frozenset([("Al", 1), ("C", 2), ("H", 6)]):  "DMA*",          # Al(CH3)2 (surface-bound or free)
    frozenset([("Al", 1), ("C", 1), ("H", 3)]):  "MMA*",          # AlCH3
    frozenset([("Al", 1)]):                        "Al*",           # bare Al adatom
    frozenset([("C", 1), ("H", 3)]):              "CH3",           # methyl radical
    frozenset([("C", 1), ("H", 4)]):              "CH4",           # methane (byproduct)
    frozenset([("Al", 2), ("C", 4), ("H", 12)]): "TMA-dimer",     # bridged dimer
    frozenset([("Al", 1), ("C", 2), ("H", 6), ("O", 1)]): "DMA-O*",  # after O-pulse
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def composition_fingerprint(atom_types: list[int]) -> frozenset:
    """Convert a list of atom type ints to a frozenset of (element, count)."""
    element_counts = Counter(ATOM_TYPE_MAP.get(t, f"T{t}") for t in atom_types)
    return frozenset(element_counts.items())


def label_species(fingerprint: frozenset) -> str:
    """Return a human-readable label for a composition fingerprint."""
    if fingerprint in KNOWN_SPECIES:
        return KNOWN_SPECIES[fingerprint]
    # Fall back to a sorted formula string
    parts = sorted(fingerprint, key=lambda x: x[0])
    return "".join(f"{el}{n}" if n > 1 else el for el, n in parts) or "empty"


def is_pure_substrate(atom_types: list[int]) -> bool:
    """True if cluster contains ONLY substrate atom types (bare slab, no adsorbate)."""
    return all(t in SUBSTRATE_TYPES for t in atom_types)


def split_cluster(atom_types: list[int]) -> tuple[list[int], list[int]]:
    """
    Split a cluster's atom types into (adsorbate_types, substrate_types).
    A mixed cluster (Al bonded to slab) contains both; we characterise
    the adsorbate fragment independently and flag it as surface-bound.
    """
    ads = [t for t in atom_types if t not in SUBSTRATE_TYPES]
    sub = [t for t in atom_types if t in SUBSTRATE_TYPES]
    return ads, sub


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def analyse_frame(
    frame_index: int,
    data: DataCollection,
    cutoff: float,
    zmax: float | None,
) -> list[dict]:
    """
    Returns a list of species records for one trajectory frame.

    Each cluster is classified as one of:
      - pure_substrate : only Si/O/substrate-H — skipped entirely
      - surface_bound  : contains both adsorbate and substrate atoms
                         (Al has bonded to the slab)
      - gas_phase      : only adsorbate atoms, floating above surface

    The species label and composition are based on adsorbate atoms only,
    so 'Al22C66H454O896Si384' becomes 'Al22C66H[n]' labelled as surface_bound.
    """
    particles = data.particles

    types       = particles["Particle Type"][:]
    pos         = particles["Position"][:]
    cluster_ids = particles["Cluster"][:]

    n_clusters = int(cluster_ids.max()) + 1
    records = []

    for cid in range(n_clusters):
        mask   = cluster_ids == cid
        ctypes = types[mask].tolist()

        # Skip bare slab clusters entirely
        if is_pure_substrate(ctypes):
            continue

        ads_types, sub_types = split_cluster(ctypes)

        # If no adsorbate atoms in this cluster (shouldn't happen after above
        # check, but guard anyway)
        if not ads_types:
            continue

        z_mean = float(pos[mask, 2].mean())
        if zmax is not None and z_mean > zmax:
            continue

        # Surface-bound: cluster contains substrate atoms AND adsorbate atoms
        is_bound = len(sub_types) > 0

        # Species label is based on adsorbate composition only
        fp    = composition_fingerprint(ads_types)
        label = label_species(fp)

        records.append({
            "frame":         frame_index,
            "species":       label,
            "binding_state": "surface_bound" if is_bound else "gas_phase",
            "cluster_size":  int(mask.sum()),
            "n_ads_atoms":   len(ads_types),
            "n_sub_atoms":   len(sub_types),
            "z_mean":        round(z_mean, 3),
            "n_Al":  sum(1 for t in ads_types if ATOM_TYPE_MAP.get(t) == "Al"),
            "n_C":   sum(1 for t in ads_types if ATOM_TYPE_MAP.get(t) == "C"),
            "n_H":   sum(1 for t in ads_types if ATOM_TYPE_MAP.get(t) == "H"),
            "n_O":   sum(1 for t in ads_types if ATOM_TYPE_MAP.get(t) == "O"),
        })

    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    dump_path: str,
    output_path: str,
    cutoff: float = DEFAULT_CUTOFF,
    stride: int = 1,
    zmax: float | None = None,
):
    print(f"Loading trajectory: {dump_path}")
    pipeline = import_file(dump_path, multiple_frames=True)

    # Insert cluster analysis modifier (runs on every frame automatically)
    pipeline.modifiers.append(
        ClusterAnalysisModifier(
            cutoff=cutoff,
            sort_by_size=False,
            compute_com=False,
        )
    )

    n_frames = pipeline.source.num_frames
    print(f"  {n_frames} frames found, stride={stride}, effective frames={len(range(0, n_frames, stride))}")
    if zmax:
        print(f"  z filter: clusters with z_mean > {zmax} Å excluded")

    all_records = []

    for i, frame in enumerate(range(0, n_frames, stride)):
        if i % 50 == 0:
            print(f"  Processing frame {frame}/{n_frames} ({100*frame//n_frames}%)...")

        data = pipeline.compute(frame)
        records = analyse_frame(frame, data, cutoff, zmax)
        all_records.extend(records)

    df = pd.DataFrame(all_records)

    # --- Aggregation 1: free gas-phase species counts per frame ---
    gas = df[df["binding_state"] == "gas_phase"]
    if not gas.empty:
        agg_gas = (
            gas.groupby(["frame", "species"])
               .size()
               .reset_index(name="count")
               .pivot(index="frame", columns="species", values="count")
               .fillna(0)
               .astype(int)
        )
    else:
        agg_gas = pd.DataFrame()

    # --- Aggregation 2: surface-bound summary per frame ---
    # Reports total Al/C bound to surface rather than giant slab formula strings.
    bound = df[df["binding_state"] == "surface_bound"]
    if not bound.empty:
        agg_bound = (
            bound.groupby("frame")
                 .agg(
                     n_bound_clusters=("species", "count"),
                     n_Al_bound=("n_Al", "sum"),
                     n_C_bound=("n_C", "sum"),
                     species_types=("species", lambda x: "|".join(sorted(x.unique())))
                 )
        )
    else:
        agg_bound = pd.DataFrame()

    # --- Save outputs ---
    raw_out   = Path(output_path).with_suffix(".raw.csv")
    gas_out   = Path(output_path)
    stem      = Path(output_path).stem
    bound_out = Path(output_path).with_name(stem + "_surface_bound.csv")

    df.to_csv(raw_out, index=False)
    if not agg_gas.empty:
        agg_gas.to_csv(gas_out)
    if not agg_bound.empty:
        agg_bound.to_csv(bound_out)

    print(f"\nDone.")
    print(f"  Raw cluster data        -> {raw_out}")
    print(f"  Gas-phase species       -> {gas_out}")
    print(f"  Surface-bound summary   -> {bound_out}")

    print(f"\nGas-phase species found: {sorted(gas['species'].unique()) if not gas.empty else []}")
    print(f"\nFinal-frame summary:")
    if not df.empty:
        last_frame = df["frame"].max()
        last = df[df["frame"] == last_frame]

        print(f"\n  [Gas phase]")
        gas_last = last[last["binding_state"] == "gas_phase"]
        if gas_last.empty:
            print("    (none)")
        else:
            for sp, grp in gas_last.groupby("species"):
                print(f"    {sp:20s}: {len(grp):3d} clusters")

        print(f"\n  [Surface bound]")
        bound_last = last[last["binding_state"] == "surface_bound"]
        if bound_last.empty:
            print("    (none)")
        else:
            print(f"    Total bound clusters : {len(bound_last)}")
            print(f"    Total Al on surface  : {int(bound_last['n_Al'].sum())}")
            print(f"    Total C on surface   : {int(bound_last['n_C'].sum())}")
            print(f"    Bound fragment types : {sorted(bound_last['species'].unique())}")

    return df, agg_gas, agg_bound


# ---------------------------------------------------------------------------
# Quick plot (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def plot_species_over_time(
    agg_gas: pd.DataFrame,
    agg_bound: pd.DataFrame,
    output_png: str = "species_over_time.png",
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    has_gas   = not agg_gas.empty
    has_bound = not agg_bound.empty
    n_panels  = int(has_gas) + int(has_bound)
    if n_panels == 0:
        print("  No data to plot")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), squeeze=False)
    axlist = axes.flatten()
    idx = 0

    if has_gas:
        ax = axlist[idx]; idx += 1
        for col in agg_gas.columns:
            ax.plot(agg_gas.index, agg_gas[col], label=col, linewidth=1.2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Cluster count")
        ax.set_title("Gas-phase species")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    if has_bound:
        ax = axlist[idx]; idx += 1
        ax.plot(agg_bound.index, agg_bound["n_Al_bound"], color="#185FA5",
                linewidth=1.4, label="Al atoms bound")
        ax.plot(agg_bound.index, agg_bound["n_C_bound"],  color="#3B6D11",
                linewidth=1.4, label="C atoms bound", linestyle="--")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Atom count")
        ax.set_title("Surface-bound adsorbate atoms over time")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    print(f"  Plot saved -> {output_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track molecular species in LAMMPS trajectories via OVITO")
    parser.add_argument("--dump",    required=True,  help="Path to LAMMPS dump file")
    parser.add_argument("--output",  required=True,  help="Output CSV path (aggregated)")
    parser.add_argument("--cutoff",  type=float, default=DEFAULT_CUTOFF,
                        help=f"Bond cutoff in Angstrom (default: {DEFAULT_CUTOFF})")
    parser.add_argument("--stride",  type=int, default=1,
                        help="Analyse every Nth frame (default: 1 = all frames)")
    parser.add_argument("--zmax",    type=float, default=None,
                        help="Ignore clusters with z_mean above this value (gas phase filter)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate a quick matplotlib plot")
    args = parser.parse_args()

    df, agg_gas, agg_bound = run(
        dump_path=args.dump,
        output_path=args.output,
        cutoff=args.cutoff,
        stride=args.stride,
        zmax=args.zmax,
    )

    if args.plot:
        plot_out = Path(args.output).with_suffix(".png")
        plot_species_over_time(agg_gas, agg_bound, str(plot_out))
