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

from lammps_common import read_lammps_dump_frames, get_lammps_types, get_lammps_ids, detect_surface_z
from site_classification import classify_substrate_sites, site_inventory_summary


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


def tail_average(df: pd.DataFrame, cols: list, n_frames: int) -> dict:
    """
    Mean +/- std over the last n_frames rows of df (by row order, i.e. the
    final n_frames analysed frames), for each column in cols. Replaces
    taking a single final-frame value (df[col].iloc[-1]), which is
    sensitive to whichever frame the trajectory happened to end on —
    e.g. the ~10-percentage-point spread seen between nominally identical
    control runs. n_frames is in analysed-frame units (i.e. after
    --stride), not raw LAMMPS timesteps or physical time.

    Returns a dict with '<col>' (mean), '<col>_std', 'n_frames_averaged',
    and 'frame_range' (min, max analysed frame actually included) so the
    averaging window used is auditable, not just trusted blindly.
    """
    tail = df.tail(n_frames)
    out: dict = {"n_frames_averaged": len(tail)}
    if "frame" in tail.columns and not tail.empty:
        out["frame_range"] = (int(tail["frame"].min()), int(tail["frame"].max()))
    else:
        out["frame_range"] = None
    for c in cols:
        if c in tail.columns:
            out[c] = float(tail[c].mean())
            out[f"{c}_std"] = float(tail[c].std()) if len(tail) > 1 else 0.0
        else:
            out[c] = None
            out[f"{c}_std"] = None
    return out


def classify_initial_sites(
    dump0_path: str,
    hydroxyl_h_type: int = 1,
    o_type: int = 2,
    si_type: int = 3,
    substrate_types: list = (1, 2, 3),
    slab_thickness: float = 3.0,
    density_fraction: float = 0.3,
    min_layer_atoms: int = 2,
    scan_bin_width: float = 1.0,
    si_o_cutoff: float = 2.0,
    o_h_cutoff: float = 1.2,
):
    """
    Classify every surface-layer O and Si atom in the pristine slab
    (dump0.lammpstrj, frame 0, before any precursor) into a reactive-site
    type (see site_classification.classify_substrate_sites for labels).

    This is the fixed reference inventory for per-site-type coverage
    fractions: unlike n_bonded_Al (which is tracked per-frame), the site
    inventory is computed once, from the pristine slab, because it's
    asking "how much of what this slab STARTED with has reacted" — the
    same role dump0/count_oh_sites plays for the existing hydroxyl-only
    coverage metric, just split by site type instead of lumped together.

    Returns
    -------
    site_by_id : dict {lammps_id: label} for every classified surface O/Si atom
    inventory  : dict {label: count} — e.g. {'hydroxyl_O': 142, 'bridging_O': 58, ...}
    sz         : the SurfaceZResult used to define the surface layer
    """
    frame_iter = read_lammps_dump_frames(dump0_path, stride=1)
    _, atoms = next(frame_iter)

    positions = atoms.get_positions()
    types = get_lammps_types(atoms)
    ids = get_lammps_ids(atoms)

    sz = detect_surface_z(
        positions, types, list(substrate_types), slab_thickness,
        density_fraction=density_fraction, min_layer_atoms=min_layer_atoms,
        scan_bin_width=scan_bin_width,
    )

    surface_mask = (positions[:, 2] <= sz.z_top) & (positions[:, 2] >= (sz.z_top - slab_thickness))

    site_by_id = classify_substrate_sites(
        positions, types, ids,
        hydroxyl_h_type=hydroxyl_h_type, o_type=o_type, si_type=si_type,
        si_o_cutoff=si_o_cutoff, o_h_cutoff=o_h_cutoff,
        atom_mask=surface_mask,
    )
    inventory = site_inventory_summary(site_by_id)
    return site_by_id, inventory, sz


def compute_site_type_coverage(al_df: pd.DataFrame, inventory: dict) -> pd.DataFrame:
    """
    Given the per-Al-atom raw records (frame, binding_state,
    bound_site_type, ...) from deposition_analysis.py (with site_by_id
    threaded through to al_bonding_analysis.analyse_al_atoms), compute
    per-frame, per-site-type consumption fractions:
        frac_<site_type>(t) = n_Al_bonded_to_<site_type>(t) / inventory[<site_type>]

    Deliberately kept SEPARATE per site type rather than summed into one
    percentage — a bonding event at a bridging_O (ring-opening) is a
    mechanistically different event from one at a hydroxyl_O, and lumping
    their counts into a single denominator would hide exactly the
    site-type-resolved picture this metric exists to show. For an
    assumption-light, single-number saturation metric, use
    surface_coverage.py's areal_density_Al instead (see project notes).
    """
    if al_df.empty or "bound_site_type" not in al_df.columns:
        return pd.DataFrame(columns=["frame"] + [f"frac_{k}" for k in inventory])

    bonded = al_df[al_df["binding_state"].isin(["chemisorbed", "incorporated"])]
    rows = []
    for frame, frame_df in bonded.groupby("frame"):
        row = {"frame": frame}
        counts = frame_df["bound_site_type"].value_counts()
        for site_type, n_initial in inventory.items():
            n_bonded_this_type = int(counts.get(site_type, 0))
            row[f"n_{site_type}"] = n_bonded_this_type
            row[f"frac_{site_type}"] = n_bonded_this_type / n_initial if n_initial > 0 else float("nan")
        rows.append(row)

    # Frames with zero bonded Al atoms are legitimately all-zero, not missing —
    # reindex over every frame present in al_df so those aren't silently dropped.
    result = pd.DataFrame(rows).set_index("frame") if rows else pd.DataFrame()
    all_frames = sorted(al_df["frame"].unique())
    if not result.empty:
        result = result.reindex(all_frames, fill_value=0).reset_index().rename(columns={"index": "frame"})
    return result


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
