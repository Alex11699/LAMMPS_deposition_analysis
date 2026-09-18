"""
snapshot.py
------------
Headless visual-validation snapshot for a single trajectory frame: plots
substrate + Al atoms color-coded by binding_state and marked by
ligand_state, with the detected z_surface drawn as a reference plane.

Designed to run on the HPC with no display (matplotlib's Agg backend),
so you can eyeball whether the classification lines up with the actual
3D structure without needing OVITO's GUI or a local copy of the dump.

Usage (standalone):
    python snapshot.py --dump dump1.lammpstrj --frame 999 --output snapshot.png

Or import render_snapshot() directly from deposition_analysis.py to reuse
the frame data already computed there (avoids re-reading the trajectory).
"""

from __future__ import annotations

import numpy as np


BINDING_COLORS = {
    "unbound":      "#d62728",   # red
    "physisorbed":  "#ff7f0e",   # orange
    "chemisorbed":  "#2ca02c",   # green
    "incorporated": "#1f77b4",   # blue
}

LIGAND_MARKERS = {
    "TMA-like": "o",
    "DMA-like": "s",
    "MMA-like": "^",
    "Al*-bare": "D",
}


def render_snapshot(
    positions: np.ndarray,
    types: np.ndarray,
    substrate_types: set,
    al_records: list,          # list of AlAtomRecord (or dicts with same fields)
    z_surface: float,
    output_path: str,
    title: str = "",
):
    """
    Two-panel snapshot: top-down (x-y) and side (x-z) views. Substrate
    atoms are small grey dots; Al atoms are colored by binding_state and
    shaped by ligand_state, with a legend for both. The side view also
    draws a horizontal line at z_surface so you can directly check
    whether "chemisorbed" Al atoms are actually sitting near the surface
    and "unbound" ones are actually far above it.
    """
    import matplotlib
    matplotlib.use("Agg")   # headless — no display needed on the HPC
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sub_mask = np.isin(types, list(substrate_types))
    sub_pos = positions[sub_mask]

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(14, 6))

    # Substrate: light grey background context in both views
    ax_top.scatter(sub_pos[:, 0], sub_pos[:, 1], s=4, c="#cccccc", alpha=0.5, label="substrate")
    ax_side.scatter(sub_pos[:, 0], sub_pos[:, 2], s=4, c="#cccccc", alpha=0.5, label="substrate")

    # z_surface reference line on the side view
    x_range = [positions[:, 0].min(), positions[:, 0].max()]
    ax_side.plot(x_range, [z_surface, z_surface], color="black", linestyle="--",
                 linewidth=1, label=f"z_surface = {z_surface:.2f} Å")

    # Al atoms, colored by binding_state, shaped by ligand_state
    for rec in al_records:
        bs = rec.binding_state if hasattr(rec, "binding_state") else rec["binding_state"]
        ls = rec.ligand_state if hasattr(rec, "ligand_state") else rec["ligand_state"]
        x = rec.x if hasattr(rec, "x") else rec["x"]
        y = rec.y if hasattr(rec, "y") else rec["y"]
        z = rec.z if hasattr(rec, "z") else rec["z"]
        color = BINDING_COLORS.get(bs, "#888888")
        marker = LIGAND_MARKERS.get(ls, "x")   # 'x' flags unexpected(n) states visually too
        ax_top.scatter(x, y, s=90, c=color, marker=marker, edgecolors="black", linewidths=0.6, zorder=5)
        ax_side.scatter(x, z, s=90, c=color, marker=marker, edgecolors="black", linewidths=0.6, zorder=5)

    ax_top.set_xlabel("x (Å)"); ax_top.set_ylabel("y (Å)")
    ax_top.set_title("Top-down view")
    ax_top.set_aspect("equal", adjustable="datalim")

    ax_side.set_xlabel("x (Å)"); ax_side.set_ylabel("z (Å)")
    ax_side.set_title("Side view")

    # Two separate legends: color = binding_state, marker shape = ligand_state
    color_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                             markeredgecolor="black", markersize=9, label=s)
                      for s, c in BINDING_COLORS.items()]
    marker_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="grey",
                              markeredgecolor="black", markersize=9, label=s)
                        for s, m in LIGAND_MARKERS.items()]
    fig.legend(handles=color_handles, title="binding_state", loc="upper left",
               bbox_to_anchor=(0.01, 0.98), fontsize=8)
    fig.legend(handles=marker_handles, title="ligand_state", loc="upper left",
               bbox_to_anchor=(0.13, 0.98), fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    from ovito.io import import_file
    from lammps_common import detect_surface_z
    from al_bonding_analysis import analyse_al_atoms

    parser = argparse.ArgumentParser(description="Render a single-frame visual validation snapshot")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--frame", type=int, default=-1, help="Frame index, -1 for last frame (default)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chem-cutoff", type=float, default=2.2)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--al-c-cutoff", type=float, default=None)
    parser.add_argument("--slab-thickness", type=float, default=3.0)
    parser.add_argument("--surface-density-fraction", type=float, default=0.3, dest="density_fraction")
    args = parser.parse_args()

    SUBSTRATE_TYPES = {1, 2, 3}
    AL_TYPE = 4
    CARBON_TYPE = 5

    pipeline = import_file(args.dump, multiple_frames=True)
    n_frames = pipeline.source.num_frames
    frame = n_frames - 1 if args.frame == -1 else args.frame
    print(f"Rendering frame {frame} of {n_frames}...")

    data = pipeline.compute(frame)
    positions = np.array(data.particles["Position"][:])
    types = np.array(data.particles["Particle Type"][:])

    sz = detect_surface_z(positions, types, list(SUBSTRATE_TYPES), args.slab_thickness,
                           density_fraction=args.density_fraction)
    al_records = analyse_al_atoms(
        data, frame, AL_TYPE, CARBON_TYPE, SUBSTRATE_TYPES,
        sz.z_surface, args.chem_cutoff, args.contact_cutoff, al_c_cutoff=args.al_c_cutoff,
    )

    render_snapshot(positions, types, SUBSTRATE_TYPES, al_records, sz.z_surface,
                     args.output, title=f"{args.dump} — frame {frame}")
    print(f"Saved -> {args.output}")
