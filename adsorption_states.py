"""
adsorption_states.py
---------------------
Pure classification logic for adsorbate fragment states, kept separate
from the OVITO pipeline machinery so it can be unit-tested against plain
numpy arrays.

State definitions (per adsorbate fragment / cluster, per frame):

  unbound       Fragment shares no cluster with any substrate atom, even
                at the loose "contact" cutoff. Floating, no interaction
                with the surface.

  physisorbed   Fragment does NOT bond to substrate at the normal
                (covalent) bonding cutoff, but DOES merge into a
                substrate-containing cluster at a looser contact cutoff
                (e.g. ~4 A, van der Waals range). In proximity, not
                chemically bonded.

  chemisorbed   Fragment merges with substrate at the normal covalent
                cutoff (a real bond has formed) AND its mean z is above
                z_surface — sitting on top of the slab, bonded.

  incorporated  Same bonding condition as chemisorbed, but mean z is at
                or below z_surface — the fragment is structurally
                embedded in the slab plane, not just perched on top.

This requires running cluster analysis twice per frame: once at the
normal (chemical) bonding cutoff, once at a looser contact cutoff.
Physisorption becomes a checkable geometric fact (touches at contact
distance but doesn't bond) rather than a z-only heuristic.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FragmentState:
    cluster_id_chem:   int      # cluster id at the chemical (bonding) cutoff
    state:              str      # 'unbound' | 'physisorbed' | 'chemisorbed' | 'incorporated'
    z_mean:              float
    n_ads_atoms:          int
    n_sub_atoms_bonded:    int    # substrate atoms in the same chem cluster (0 unless chemisorbed/incorporated)


def classify_fragments(
    types: np.ndarray,
    positions: np.ndarray,
    chem_cluster_ids: np.ndarray,
    contact_cluster_ids: np.ndarray,
    substrate_types: set,
    z_surface: float,
) -> list[FragmentState]:
    """
    Classify every adsorbate fragment (cluster at the chemical cutoff) into
    one of unbound / physisorbed / chemisorbed / incorporated.

    Parameters
    ----------
    types : (N,) raw LAMMPS type integers, all atoms in the frame
    positions : (N, 3) Cartesian positions, all atoms in the frame
    chem_cluster_ids : (N,) cluster id per atom at the bonding cutoff
    contact_cluster_ids : (N,) cluster id per atom at the looser contact cutoff
    substrate_types : set of LAMMPS type integers that count as substrate
    z_surface : robust surface z-reference (from detect_surface_z)

    Returns one FragmentState per adsorbate cluster found at the chemical
    cutoff (pure-substrate clusters are skipped entirely).
    """
    sub_mask = np.isin(types, list(substrate_types))
    ads_mask = ~sub_mask

    results = []
    chem_ids = np.unique(chem_cluster_ids[ads_mask])

    for cid in chem_ids:
        frag_mask = (chem_cluster_ids == cid) & ads_mask
        if not frag_mask.any():
            continue

        n_ads_atoms = int(frag_mask.sum())
        z_mean = float(positions[frag_mask, 2].mean())

        # Chemisorption test: does this chem cluster (fragment + anything
        # bonded to it) include any substrate atoms?
        full_chem_cluster_mask = (chem_cluster_ids == cid)
        n_sub_bonded = int((full_chem_cluster_mask & sub_mask).sum())

        if n_sub_bonded > 0:
            state = "incorporated" if z_mean <= z_surface else "chemisorbed"
        else:
            # Not chemically bonded — check contact-cutoff clusters for
            # this same fragment to see if it's merely in proximity.
            contact_ids_here = np.unique(contact_cluster_ids[frag_mask])
            is_physisorbed = False
            for ccid in contact_ids_here:
                contact_cluster_mask = (contact_cluster_ids == ccid)
                if (contact_cluster_mask & sub_mask).any():
                    is_physisorbed = True
                    break
            state = "physisorbed" if is_physisorbed else "unbound"

        results.append(FragmentState(
            cluster_id_chem=int(cid),
            state=state,
            z_mean=z_mean,
            n_ads_atoms=n_ads_atoms,
            n_sub_atoms_bonded=n_sub_bonded,
        ))

    return results
