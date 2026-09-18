"""
site_classification.py
-----------------------
Classifies substrate O and Si atoms into reactive-site types by direct
coordination counting (Si-O and O-H distances), independent of the
cluster-analysis machinery in adsorption_states.py.

Why this is separate from adsorption_states.py: that module answers "is
this adsorbate fragment bonded to the substrate" using whole-cluster
connectivity. This module answers a different question — "what KIND of
substrate site is this specific O/Si atom" — using only pairwise
distances, the same way al_bonding_analysis.py determines Al ligand
counts directly rather than via cluster membership.

Site labels (surface O/Si atoms only):
    hydroxyl_O            O bonded to exactly 1 Si and 1 H  (-OH site)
    bridging_O             O bonded to 2 Si and 0 H            (Si-O-Si siloxane bridge)
    dangling_O              O bonded to <=1 Si and 0 H           (undercoordinated O)
    other_O                  anything else (e.g. already Al-bonded, or >2 Si)
    coordinated_Si            Si bonded to >=4 O                    (bulk-like)
    undercoordinated_Si         Si bonded to <4 O                      (dangling-bond Si)

IMPORTANT — validate the cutoffs before trusting results:
si_o_cutoff and o_h_cutoff are geometric guesses (2.0 A, 1.2 A) until
checked against this slab's own Si-O / O-H distance distribution. Build
a distance histogram from the pristine dump0 slab (same data-driven
approach as al_c_distance_histogram.find_bonding_cutoff, which found a
genuine trough for Al-C at ~2.4-2.45 A) before relying on classifications
near the boundary. Expect some fraction of near-cutoff atoms to be
genuinely ambiguous (thermal flicker) — this mirrors the "unexpected(n)"
ligand states already handled in al_bonding_analysis.py.

This module does NOT distinguish strained (reactive) siloxane bridges
from relaxed (inert) ones — that needs Si-O-Si bond-angle analysis on
top of coordination counting, deliberately left out of this first pass.
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from scipy.spatial import cKDTree


def classify_substrate_sites(
    positions: np.ndarray,
    types: np.ndarray,
    ids: np.ndarray,
    hydroxyl_h_type: int = 1,
    o_type: int = 2,
    si_type: int = 3,
    si_o_cutoff: float = 2.0,
    o_h_cutoff: float = 1.2,
    atom_mask: np.ndarray | None = None,
) -> dict:
    """
    Classify every O and Si atom (optionally restricted to atom_mask,
    e.g. "in the surface layer") into a site label, by direct-distance
    coordination counting.

    Parameters
    ----------
    positions : (N, 3) Cartesian positions, all atoms in the frame
    types : (N,) raw LAMMPS type integers, all atoms in the frame
    ids : (N,) LAMMPS atom ids, all atoms in the frame — used as the
          returned dict's keys, since array index isn't a stable identity
          across frames/files (see lammps_common.get_lammps_ids)
    hydroxyl_h_type, o_type, si_type : raw LAMMPS type integers
    si_o_cutoff, o_h_cutoff : bonding cutoffs in Angstrom — VALIDATE
        against this slab's own distance histogram before trusting
        (see module docstring)
    atom_mask : (N,) bool, restrict classification to these atoms (e.g.
                the surface layer from detect_surface_z). None = all
                O/Si atoms in the frame.

    Returns
    -------
    dict {lammps_id: label}, one entry per classified O/Si atom.
    """
    o_idx = np.where(types == o_type)[0]
    si_idx = np.where(types == si_type)[0]
    h_idx = np.where(types == hydroxyl_h_type)[0]

    si_tree = cKDTree(positions[si_idx]) if len(si_idx) else None
    h_tree = cKDTree(positions[h_idx]) if len(h_idx) else None
    o_tree = cKDTree(positions[o_idx]) if len(o_idx) else None

    site_by_id: dict = {}

    target_o = o_idx if atom_mask is None else o_idx[atom_mask[o_idx]]
    for i in target_o:
        n_si = len(si_tree.query_ball_point(positions[i], si_o_cutoff)) if si_tree else 0
        n_h = len(h_tree.query_ball_point(positions[i], o_h_cutoff)) if h_tree else 0
        if n_si == 1 and n_h == 1:
            label = "hydroxyl_O"
        elif n_si == 2 and n_h == 0:
            label = "bridging_O"
        elif n_si <= 1 and n_h == 0:
            label = "dangling_O"
        else:
            label = "other_O"
        site_by_id[int(ids[i])] = label

    target_si = si_idx if atom_mask is None else si_idx[atom_mask[si_idx]]
    for i in target_si:
        n_o = len(o_tree.query_ball_point(positions[i], si_o_cutoff)) if o_tree else 0
        site_by_id[int(ids[i])] = "coordinated_Si" if n_o >= 4 else "undercoordinated_Si"

    return site_by_id


def site_inventory_summary(site_by_id: dict) -> dict:
    """Count of atoms per site label — the denominator for per-site-type coverage fractions."""
    return dict(Counter(site_by_id.values()))
