"""
lammps_common.py
-----------------
Shared, validated logic for analysing LAMMPS TMA/SiO2 deposition
trajectories. Extracted from surface_coverage.py after debugging against
real trajectory data, so both surface_coverage.py and deposition_analysis.py
(species + coverage + adsorption-state master script) use the same
tested code rather than two copies that can drift apart.

Contents:
  - Robust density-based z_surface / z_top detection (SurfaceZResult,
    detect_surface_z, format_z_histogram)
  - ASE-based LAMMPS dump reader with raw type extraction and correct
    Cartesian coordinates (read_lammps_dump_frames, get_lammps_types,
    get_cell_area) — for scripts that don't already have OVITO's
    DataCollection arrays available.

Both detect_surface_z and format_z_histogram operate on plain numpy
arrays (positions, types), so they work equally well fed from ASE
(surface_coverage.py) or from an OVITO DataCollection's
particles["Position"] / particles["Particle Type"] arrays
(deposition_analysis.py) — no need to read the trajectory twice with
two different libraries just to reuse this logic.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# LAMMPS dump reader (ASE-based) — raw types, correct Cartesian coordinates
# ---------------------------------------------------------------------------

def _parse_dump_header(dump_path: str) -> dict:
    """
    Read the first frame header of a LAMMPS dump file and return:
      - 'scaled': True if coordinates are xs/ys/zs (fractional)
      - 'type_col': 0-based index of the 'type' column in the ATOMS block
      - 'columns': full list of column names from ITEM: ATOMS
    """
    with open(dump_path) as f:
        for line in f:
            if line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]   # strip "ITEM:" and "ATOMS"
                return {
                    "scaled":   any(c in ("xs", "ys", "zs") for c in cols),
                    "type_col": cols.index("type") if "type" in cols else None,
                    "columns":  cols,
                }
    raise ValueError(f"No 'ITEM: ATOMS' line found in {dump_path}")


def _iter_raw_type_arrays(dump_path: str, type_col: int):
    """
    Generator yielding one np.ndarray of raw LAMMPS integer types per frame,
    parsed directly from the dump text (bypasses ASE's atomic-number
    remapping, where LAMMPS type 4 would otherwise become atomic number 4
    = Be instead of staying 4).
    """
    with open(dump_path) as f:
        n_atoms = None
        in_atoms = False
        rows = []
        for line in f:
            line = line.rstrip()
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = None
                in_atoms = False
                rows = []
                continue
            if n_atoms is None and line.isdigit():
                n_atoms = int(line)
                continue
            if line.startswith("ITEM: ATOMS"):
                in_atoms = True
                rows = []
                continue
            if in_atoms and n_atoms is not None:
                rows.append(int(line.split()[type_col]))
                if len(rows) == n_atoms:
                    yield np.array(rows, dtype=int)
                    in_atoms = False
                    n_atoms = None
                    rows = []


def read_lammps_dump_frames(dump_path: str, stride: int = 1):
    """
    Generator yielding (frame_index, ase.Atoms) for every Nth frame.

    - Positions are Cartesian Å. ASE's lammps-dump-text reader already
      converts scaled (xs/ys/zs) coordinates to Cartesian correctly,
      including for triclinic boxes — no additional conversion is applied
      (a prior version double-applied the cell matrix and produced
      nonsensical coordinates hundreds of Å outside the box; don't
      reintroduce that).
    - atoms.arrays['type'] holds raw LAMMPS integer type IDs, parsed
      directly from the dump text and injected before yielding, bypassing
      ASE's atomic-number remapping.
    - Cell geometry, box bounds, and PBC flags come from ASE (correct).
    """
    from ase.io import iread

    header = _parse_dump_header(dump_path)
    type_col = header["type_col"]
    if type_col is None:
        raise ValueError(
            "No 'type' column found in dump ITEM: ATOMS header. "
            f"Columns present: {header['columns']}"
        )

    raw_type_gen = _iter_raw_type_arrays(dump_path, type_col)

    for i, atoms in enumerate(iread(dump_path, format="lammps-dump-text", index=":")):
        raw_types = next(raw_type_gen)
        if i % stride == 0:
            atoms.arrays["type"] = raw_types
            yield i, atoms


def get_lammps_types(atoms) -> np.ndarray:
    """
    Return the raw LAMMPS integer type column (1-indexed) for every atom.
    read_lammps_dump_frames() always injects these into atoms.arrays['type']
    directly from the dump text, bypassing ASE's atomic-number remapping.
    """
    if "type" in atoms.arrays:
        return np.array(atoms.arrays["type"], dtype=int)
    raise KeyError(
        "atoms.arrays['type'] not found. Was this frame read by "
        "read_lammps_dump_frames()? Available keys: "
        f"{list(atoms.arrays.keys())}"
    )


def get_cell_area(atoms) -> float:
    """
    Return the xy cross-sectional area of the simulation cell in Angstrom^2.
    Works for orthorhombic and triclinic boxes (non-zero xy xz yz tilts).
    """
    cell = atoms.get_cell()
    a, b = cell[0], cell[1]
    return float(np.linalg.norm(np.cross(a, b)))


# ---------------------------------------------------------------------------
# Robust density-based surface z-reference detection
# ---------------------------------------------------------------------------

@dataclass
class SurfaceZResult:
    """Diagnostics for one frame's auto-detected surface reference."""
    z_surface:           float  # mean z of the detected surface layer
    z_top:                float  # robust top-of-slab z (density-thresholded)
    z_top_absolute:       float  # literal highest substrate atom z, pre-rejection
    z_std:                 float  # roughness of the detected layer
    n_layer_atoms:         int    # substrate atoms used to compute z_surface
    n_substrate_total:     int    # total substrate atoms in the frame
    n_stray_atoms:         int    # substrate atoms above z_top, excluded
    bulk_density:          float  # densest scan_bin_width slice found
    density_threshold:     float  # atom-count threshold actually applied
    z_bin_edges:            np.ndarray
    z_bin_counts:            np.ndarray


def detect_surface_z(positions: np.ndarray, types: np.ndarray,
                     substrate_types: list, slab_thickness: float,
                     density_fraction: float = 0.3,
                     min_layer_atoms: int = 2,
                     scan_bin_width: float = 1.0) -> SurfaceZResult:
    """
    Find the z-coordinate of the topmost substrate layer, robust to a small
    number of stray substrate atoms (e.g. a desorbed byproduct that still
    carries a substrate atom type) without misclassifying genuine surface
    roughness as stray.

    Builds a z-density histogram of substrate atoms, takes the densest
    slice as a proxy for "bulk" density, and scans down from the absolute
    top for the first slice whose count reaches density_fraction *
    bulk_density (with min_layer_atoms as an absolute floor). That slice's
    upper edge becomes z_top; substrate atoms above it are reported as
    strays. z_surface is the mean z of substrate atoms within
    slab_thickness of z_top.

    Validated against a real ~1500-atom amorphous SiO2 slab where a fixed
    absolute atom-count threshold failed (either latched onto stray atoms,
    or dug through legitimate surface roughness into the bulk); this
    relative-density approach self-scales to slab size/thickness instead.
    """
    sub_mask = np.isin(types, substrate_types)
    n_sub_total = int(sub_mask.sum())
    if n_sub_total == 0:
        raise ValueError("No substrate atoms found — check substrate_types")

    sub_z = positions[sub_mask, 2]
    z_top_absolute = float(sub_z.max())
    z_bottom = float(sub_z.min())

    n_bins = max(1, int(np.ceil((z_top_absolute - z_bottom) / scan_bin_width)))
    edges = z_top_absolute - scan_bin_width * np.arange(n_bins + 1)
    edges = edges[::-1]
    counts, edges = np.histogram(sub_z, bins=edges)

    bulk_density = float(counts.max()) if counts.size else 0.0
    density_threshold = max(min_layer_atoms, density_fraction * bulk_density)

    z_top = z_top_absolute
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] >= density_threshold:
            z_top = float(edges[i + 1])
            break

    n_stray_atoms = int((sub_z > z_top).sum())

    surface_layer_mask = (sub_z <= z_top) & (sub_z >= (z_top - slab_thickness))
    layer_z = sub_z[surface_layer_mask]

    return SurfaceZResult(
        z_surface=float(layer_z.mean()),
        z_top=z_top,
        z_top_absolute=z_top_absolute,
        z_std=float(layer_z.std()) if layer_z.size > 1 else 0.0,
        n_layer_atoms=int(layer_z.size),
        n_substrate_total=n_sub_total,
        n_stray_atoms=n_stray_atoms,
        bulk_density=bulk_density,
        density_threshold=density_threshold,
        z_bin_edges=edges,
        z_bin_counts=counts,
    )


def format_z_histogram(edges: np.ndarray, counts: np.ndarray, z_top: float,
                        bar_width: int = 50) -> str:
    """
    Render a substrate z-density histogram as text, top to bottom, with the
    chosen z_top marked — lets you visually verify the detection against
    the real density profile instead of trusting it blindly.
    """
    lines = []
    max_count = int(counts.max()) if counts.size else 1
    for i in range(len(counts) - 1, -1, -1):
        lo, hi, c = edges[i], edges[i + 1], counts[i]
        bar = "#" * max(0, round(bar_width * c / max_count)) if max_count > 0 else ""
        marker = "  <- z_top" if abs(hi - z_top) < 1e-9 else ""
        lines.append(f"      [{lo:8.2f}, {hi:8.2f}) {c:5d}  {bar}{marker}")
    return "\n".join(lines)
