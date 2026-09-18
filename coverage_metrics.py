"""
coverage_metrics.py
----------------------
Surface coverage % and sticking coefficient, computed entirely from
trajectory data — no external literature densities or dosing logs needed.

Coverage %:
    denominator = number of hydroxyl-H (LAMMPS type 1, by default) atoms
    in the surface layer of the FRESH slab (dump0.lammpstrj, frame 0,
    before any precursor) — the real number of reactive -OH sites this
    specific slab started with.
    numerator = Al atoms currently chemisorbed or incorporated.
    coverage_pct(t) = 100 * n_bonded_Al(t) / n_oh_sites_initial

Sticking coefficient:
    fix deposit inserts one whole TMA molecule (one Al atom) per
    successful attempt, so the cumulative count of unique Al lammps_ids
    that have EVER appeared in the trajectory up to frame t is exactly
    the cumulative number of precursor molecules delivered — robust to
    failed insertion attempts (LAMMPS "near" keyword can reject an
    attempt if the target region is occupied), unlike trusting the
    nominal N/M schedule from the input script.
    sticking(t) = n_bonded_Al(t) / cumulative_unique_Al_seen(t)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lammps_common import read_lammps_dump_frames, get_lammps_types, detect_surface_z


def count_oh_sites(
    dump0_path: str,
    hydroxyl_type: int = 1,
    substrate_types: list = (1, 2, 3),
    slab_thickness: float = 3.0,
    density_fraction: float = 0.3,
    min_layer_atoms: int = 2,
    scan_bin_width: float = 1.0,
) -> int:
    """
    Count hydroxyl-H atoms in the surface layer of frame 0 of the pure-slab
    dump (dump0.lammpstrj) — the initial reactive-site count for this slab.
    """
    frame_iter = read_lammps_dump_frames(dump0_path, stride=1)
    _, atoms = next(frame_iter)

    positions = atoms.get_positions()
    types = get_lammps_types(atoms)

    sz = detect_surface_z(
        positions, types, list(substrate_types), slab_thickness,
        density_fraction=density_fraction, min_layer_atoms=min_layer_atoms,
        scan_bin_width=scan_bin_width,
    )

    oh_mask = (types == hydroxyl_type) & \
              (positions[:, 2] <= sz.z_top) & (positions[:, 2] >= (sz.z_top - slab_thickness))
    n_oh = int(oh_mask.sum())
    return n_oh, sz


def compute_coverage_sticking(al_df: pd.DataFrame, n_oh_sites: int) -> pd.DataFrame:
    """
    Given the per-Al-atom raw records (frame, lammps_id, binding_state, ...)
    from deposition_analysis.py, compute per-frame coverage % and sticking
    coefficient.
    """
    if al_df.empty:
        return pd.DataFrame(columns=["frame", "n_bonded", "cumulative_Al_seen",
                                       "coverage_pct", "sticking_coefficient"])

    frames = sorted(al_df["frame"].unique())
    rows = []
    seen_ids = set()

    for frame in frames:
        frame_df = al_df[al_df["frame"] == frame]
        seen_ids.update(frame_df["lammps_id"].tolist())

        n_bonded = int(frame_df["binding_state"].isin(["chemisorbed", "incorporated"]).sum())
        cumulative = len(seen_ids)

        coverage_pct = 100.0 * n_bonded / n_oh_sites if n_oh_sites > 0 else float("nan")
        sticking = n_bonded / cumulative if cumulative > 0 else float("nan")

        rows.append({
            "frame": frame,
            "n_bonded": n_bonded,
            "cumulative_Al_seen": cumulative,
            "coverage_pct": round(coverage_pct, 3),
            "sticking_coefficient": round(sticking, 4),
        })

    return pd.DataFrame(rows)
