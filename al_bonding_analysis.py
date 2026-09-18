"""
al_bonding_analysis.py
------------------------
Per-metal-atom (Al) local bonding analysis, computed directly from
distances via OVITO's CutoffNeighborFinder — NOT from whole-cluster
connectivity.

Why this exists separately from adsorption_states.py's cluster-based
classification: at moderate-to-high surface coverage, individual
chemisorbed TMA-derived fragments start bonding to EACH OTHER through
the substrate (Al-O-Al bridging is real ALD film chemistry), so cluster
analysis correctly merges them into one large connected network. That's
correct for asking "is this atom touching the slab at all" but wrong for
asking "how many methyl ligands does THIS SPECIFIC Al atom still carry" —
a whole-network composition string like Al36C83H150 answers neither
question usefully.

For TMA/SiO2 ALD specifically, ligand count on an individual Al atom is
the physically meaningful reaction-progress indicator:
    3 C ligands -> intact TMA, unreacted
    2 C ligands -> DMA* (lost one ligand via e.g. -CH4 release) —
                   the textbook "properly chemisorbed, ready for the
                   next half-cycle" state for TMA/SiO2 ALD
    1 C ligand  -> MMA* (lost two ligands)
    0 C ligands -> Al* (fully stripped, e.g. incorporated/bridging Al)

Binding state (chemisorbed / physisorbed / incorporated / unbound) is
determined by the DISTANCE from this specific Al atom to its nearest
substrate atom — not by whether it shares a cluster with one — so an Al
atom 15 atoms deep in a percolated network isn't misreported as
"chemisorbed" just because something else in that network happens to
touch the slab far away.

All raw distances and neighbor counts are recorded alongside the derived
categorical labels, per your request to keep several distance-based
metrics available rather than collapsing straight to one threshold call —
the categorical columns can be recomputed with different cutoffs later
without rerunning the (expensive) trajectory pass.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from ovito.data import CutoffNeighborFinder


@dataclass
class AlAtomRecord:
    frame:                 int
    particle_index:         int
    lammps_id:               int
    x: float; y: float; z: float
    z_surface:                float
    z_rel:                     float   # z - z_surface (positive = above surface)

    n_C_ligands:               int     # C atoms within al_c_cutoff (methyl ligands still attached)
    ligand_state:               str     # 'TMA-like' | 'DMA-like' | 'MMA-like' | 'Al*-bare' | 'unexpected(n)'

    nearest_substrate_dist:      float   # min distance to any substrate atom within scan range (NaN if none found)
    nearest_substrate_id:          int | None  # LAMMPS id of that nearest substrate atom (None if none found)
    n_substrate_neighbors:        dict    # {cutoff: count} for each scanned cutoff — multi-scale coordination
    binding_state:                 str     # 'unbound' | 'physisorbed' | 'chemisorbed' | 'incorporated'
    bound_site_type:                str | None  # site_classification label of nearest_substrate_id, if
                                                 # binding_state is chemisorbed/incorporated AND site_by_id
                                                 # was supplied; else None. Looked up from the PRISTINE
                                                 # (dump0) site classification, so this answers "what kind
                                                 # of site did this Al originally react with" even after
                                                 # the local chemistry there has since changed (e.g. its H
                                                 # left as CH4) — not "what does this atom look like now".


def analyse_al_atoms(
    data,
    frame_index: int,
    al_type: int,
    carbon_type: int,
    substrate_types: set,
    z_surface: float,
    chem_cutoff: float,
    contact_cutoff: float,
    al_c_cutoff: float | None = None,
    scan_cutoffs: list[float] | None = None,
    site_by_id: dict | None = None,
) -> list[AlAtomRecord]:
    """
    For every Al atom in this frame, compute local ligand count and
    substrate-bonding distance directly (not via cluster connectivity).

    Parameters
    ----------
    data : ovito.data.DataCollection for this frame (already computed)
    al_type, carbon_type : raw LAMMPS type integers
    substrate_types : set of LAMMPS type integers counted as substrate
    z_surface : robust surface z-reference (from lammps_common.detect_surface_z)
    chem_cutoff : distance defining a real Al-substrate bond (chemisorption)
    contact_cutoff : looser distance defining physical contact (physisorption)
    al_c_cutoff : distance defining an intact Al-C ligand bond.
                  Defaults to chem_cutoff if not given.
    scan_cutoffs : list of distances to report substrate-neighbor counts at,
                   for inspecting coordination at multiple scales. Defaults
                   to [2.0, chem_cutoff, 3.0, contact_cutoff].
    site_by_id : optional {lammps_id: label} from
                 coverage_metrics.classify_initial_sites (pristine dump0
                 classification). If given, each chemisorbed/incorporated
                 Al is tagged with the site type of its nearest substrate
                 neighbor (bound_site_type). If omitted, bound_site_type
                 is None for every record — this is purely opt-in and
                 doesn't change behaviour otherwise.
    """
    if al_c_cutoff is None:
        al_c_cutoff = chem_cutoff
    if scan_cutoffs is None:
        scan_cutoffs = sorted(set([2.0, chem_cutoff, 3.0, contact_cutoff]))

    max_cutoff = max(contact_cutoff, al_c_cutoff, max(scan_cutoffs))
    finder = CutoffNeighborFinder(max_cutoff, data)

    types = np.array(data.particles["Particle Type"][:])
    positions = np.array(data.particles["Position"][:])
    ids = np.array(data.particles["Particle Identifier"][:]) if "Particle Identifier" in data.particles else np.arange(len(types))

    al_indices = np.where(types == al_type)[0]
    records = []

    for idx in al_indices:
        pos = positions[idx]

        sub_dists = []
        sub_dist_ids = []   # LAMMPS ids parallel to sub_dists, for nearest-neighbor site lookup
        n_at_cutoff = {c: 0 for c in scan_cutoffs}
        n_c_ligands = 0

        for neigh in finder.find(idx):
            ntype = types[neigh.index]
            d = neigh.distance
            if ntype in substrate_types:
                sub_dists.append(d)
                sub_dist_ids.append(int(ids[neigh.index]))
                for c in scan_cutoffs:
                    if d <= c:
                        n_at_cutoff[c] += 1
            if ntype == carbon_type and d <= al_c_cutoff:
                # A methyl carbon is chemically attached to exactly one Al, so
                # only count it here if THIS Al is its nearest Al overall — not
                # merely "within cutoff". Without this check, two Al centers
                # closer together than 2*al_c_cutoff can each independently
                # count the same shared carbon, inflating both ligand counts
                # (confirmed against real trajectory data: several
                # "unexpected(4)" atoms turned out to be genuine DMA-like
                # atoms that had borrowed a neighboring Al's carbon).
                c_idx = neigh.index
                is_nearest_al = True
                for c_neigh in finder.find(c_idx):
                    if (types[c_neigh.index] == al_type
                            and c_neigh.index != idx
                            and c_neigh.distance < d - 1e-9):
                        is_nearest_al = False
                        break
                if is_nearest_al:
                    n_c_ligands += 1

        if sub_dists:
            nearest_i = int(np.argmin(sub_dists))
            nearest_sub_dist = sub_dists[nearest_i]
            nearest_sub_id = sub_dist_ids[nearest_i]
        else:
            nearest_sub_dist = float("nan")
            nearest_sub_id = None

        # Ligand state from direct C-neighbor count
        ligand_state = {3: "TMA-like", 2: "DMA-like", 1: "MMA-like", 0: "Al*-bare"}.get(
            n_c_ligands, f"unexpected({n_c_ligands})"
        )

        # Binding state from direct nearest-substrate distance (NOT cluster membership)
        z_rel = pos[2] - z_surface
        if not sub_dists or nearest_sub_dist > contact_cutoff:
            binding_state = "unbound"
        elif nearest_sub_dist > chem_cutoff:
            binding_state = "physisorbed"
        else:
            binding_state = "incorporated" if z_rel <= 0 else "chemisorbed"

        bound_site_type = None
        if site_by_id is not None and binding_state in ("chemisorbed", "incorporated") and nearest_sub_id is not None:
            # Falls back to None (not KeyError) for a substrate atom outside
            # the classified surface layer, or outside the pristine dump0
            # slab entirely (shouldn't happen for substrate atoms, but the
            # classification is opt-in and shouldn't crash the whole run).
            bound_site_type = site_by_id.get(nearest_sub_id)

        records.append(AlAtomRecord(
            frame=frame_index,
            particle_index=int(idx),
            lammps_id=int(ids[idx]),
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
            z_surface=float(z_surface),
            z_rel=float(z_rel),
            n_C_ligands=n_c_ligands,
            ligand_state=ligand_state,
            nearest_substrate_dist=float(nearest_sub_dist),
            nearest_substrate_id=nearest_sub_id,
            n_substrate_neighbors=dict(n_at_cutoff),
            binding_state=binding_state,
            bound_site_type=bound_site_type,
        ))

    return records
