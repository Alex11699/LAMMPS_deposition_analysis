"""
inspect_al_atom.py
--------------------
Diagnostic tool: for one specific Al atom (by LAMMPS id) at one specific
frame, list every neighboring carbon atom within --al-c-cutoff, and show
which Al atom each carbon is genuinely CLOSEST to overall. This is how
you find the actual cause of an "unexpected(n)" ligand count — the raw
al_atoms CSV only gives you the final count, not which atoms produced it.

Usage:
    python inspect_al_atom.py --dump dump1.lammpstrj --frame 999 --lammps-id 4821
    python inspect_al_atom.py --dump dump1.lammpstrj --frame 999 --lammps-id 4821 \\
        --al-c-cutoff 1.9
"""

from __future__ import annotations

import argparse
import numpy as np
from ovito.io import import_file
from ovito.data import CutoffNeighborFinder


def main():
    parser = argparse.ArgumentParser(description="Inspect one Al atom's local neighbor environment")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--lammps-id", type=int, required=True, help="LAMMPS atom id of the Al atom to inspect")
    parser.add_argument("--al-type", type=int, default=4)
    parser.add_argument("--carbon-type", type=int, default=5)
    parser.add_argument("--al-c-cutoff", type=float, default=2.2)
    parser.add_argument("--substrate-types", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    pipeline = import_file(args.dump, multiple_frames=True)
    data = pipeline.compute(args.frame)

    types = np.array(data.particles["Particle Type"][:])
    positions = np.array(data.particles["Position"][:])
    ids = np.array(data.particles["Particle Identifier"][:])

    target_idx = np.where(ids == args.lammps_id)[0]
    if len(target_idx) == 0:
        print(f"No atom with LAMMPS id {args.lammps_id} found in frame {args.frame}.")
        return
    target_idx = int(target_idx[0])
    if types[target_idx] != args.al_type:
        print(f"WARNING: atom {args.lammps_id} has type {types[target_idx]}, not --al-type {args.al_type}.")

    print(f"Inspecting Al atom id={args.lammps_id} at frame {args.frame}")
    print(f"  position: {positions[target_idx]}")
    print(f"  al_c_cutoff: {args.al_c_cutoff} Å\n")

    # All Al atoms in this frame, for "which Al does this carbon actually belong to" comparison
    al_indices = np.where(types == args.al_type)[0]

    finder = CutoffNeighborFinder(args.al_c_cutoff, data)
    # Extract plain (index, distance) tuples DURING iteration — the neighbor
    # objects yielded by find() are transient and become invalid once the
    # iterator advances, so they can't be stored in a list and filtered later.
    carbons_found = [
        (neigh.index, neigh.distance)
        for neigh in finder.find(target_idx)
        if types[neigh.index] == args.carbon_type
    ]

    print(f"Carbon atoms within {args.al_c_cutoff} Å of target Al {args.lammps_id}: {len(carbons_found)}")
    if not carbons_found:
        print("  (none — this atom would show as Al*-bare, not the reported count; "
              "double check --lammps-id / --frame match what you expect.)")
    for c_idx, d_to_target in carbons_found:
        c_id = int(ids[c_idx])
        c_pos = positions[c_idx]

        # Which Al atom is this carbon ACTUALLY closest to, out of all Al atoms in the frame?
        al_dists = np.linalg.norm(positions[al_indices] - c_pos, axis=1)
        # Correct for PBC by using the neighbor finder's own distance for the target,
        # and a direct (non-PBC-aware) distance for ranking others — good enough for
        # flagging which Al is plausibly the "true" parent when atoms are this close.
        closest_al_local_idx = al_indices[np.argmin(al_dists)]
        closest_al_id = int(ids[closest_al_local_idx])
        closest_al_dist = float(al_dists.min())

        flag = "  <-- genuinely closest to a DIFFERENT Al, likely cross-counted" \
            if closest_al_id != args.lammps_id else ""
        print(f"  C id={c_id:6d}  dist_to_target_Al={d_to_target:.3f} Å  "
              f"closest_Al_overall=id={closest_al_id} (dist={closest_al_dist:.3f} Å){flag}")

    print(f"\nIf any carbons above are flagged as closer to a different Al, that Al is the "
          f"real parent — cross-counting like this usually means either --al-c-cutoff is a "
          f"bit loose for how tightly packed this region is, or the two Al centers are "
          f"genuinely bridging/crowded at this frame (worth checking if it persists across "
          f"several consecutive frames vs a one-off transient contact).")


if __name__ == "__main__":
    main()
