"""
surface_coverage.py
-------------------
Compute surface coverage and z-density profiles from LAMMPS dump files.
Uses ASE for trajectory reading and NumPy for all heavy lifting —
no GUI required, runs headlessly on HPC.

Outputs:
  - coverage.csv     : adsorbate atom count + areal density per frame
  - z_profile.csv    : time-averaged z-density histogram
  - coverage.png     : coverage vs time plot (if --plot)
  - z_profile.png    : z-density heatmap over time (if --plot)

Usage:
    python surface_coverage.py --dump traj.dump --output coverage.csv
    python surface_coverage.py --dump traj.dump --output coverage.csv \\
        --adsorbate-types 1 2 3 --substrate-types 4 5 \\
        --surface-z-ref auto --slab-thickness 3.0 --band-width 8.0 \\
        --stride 50 --plot

Requires: ase, numpy, pandas
Install:  pip install ase numpy pandas
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration defaults — edit to match your system
# ---------------------------------------------------------------------------

# LAMMPS atom types for adsorbate atoms (TMA-derived: Al, C, H_methyl)
DEFAULT_ADSORBATE_TYPES = [4, 5, 6]   # Al, C, H(methyl)

# LAMMPS atom types for substrate (H_hydroxyl, O, Si)
DEFAULT_SUBSTRATE_TYPES = [1, 2, 3]   # H(substrate), O, Si

# Surface reference: 'auto' detects from topmost substrate atom each frame,
# or provide a fixed float (Angstrom) if your slab doesn't drift.
DEFAULT_SURFACE_Z_REF = "auto"

# How far below the topmost substrate atom still counts as "slab surface"
# (used when auto-detecting z_surface)
DEFAULT_SLAB_THICKNESS = 3.0   # Angstrom

# Width of the adsorbate band ABOVE z_surface to count as "on surface"
DEFAULT_BAND_WIDTH = 8.0       # Angstrom

# Depth of the adsorbate band BELOW z_surface to count as "on surface".
# Set > 0 to capture adsorbate atoms that have incorporated into the slab
# (reacted and bonded below the mean surface plane). 0 = above-surface only.
DEFAULT_BAND_DEPTH = 0.0       # Angstrom

# Z histogram bin width for density profiles
Z_BIN_WIDTH = 0.5              # Angstrom


# ---------------------------------------------------------------------------
# LAMMPS dump reader (pure ASE)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LAMMPS dump reader
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
    Generator that yields one np.ndarray of raw LAMMPS integer types per
    frame, by scanning the dump file for ITEM: NUMBER OF ATOMS / ITEM: ATOMS
    blocks and reading only the type column.

    This bypasses ASE's atomic-number remapping entirely — the integers
    here are exactly what LAMMPS wrote, so LAMMPS type 4 stays 4, not 13.
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

    - Positions are Cartesian Å. ASE's lammps-dump-text reader converts
      scaled (xs/ys/zs) coordinates to Cartesian correctly, including for
      triclinic boxes — we do NOT apply any additional conversion.
    - atoms.arrays['type'] holds raw LAMMPS integer type IDs, parsed
      directly from the dump text and injected before yielding, bypassing
      ASE's atomic-number remapping (LAMMPS type 4 stays 4, not Be).
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
            # ASE already converted xs/ys/zs → Cartesian Å. No extra step needed.
            # Inject raw LAMMPS types, overwriting ASE's atomic-number remapping.
            atoms.arrays["type"] = raw_types
            yield i, atoms
        # else: consume raw_types to keep generator in sync, but don't yield


def get_cell_area(atoms) -> float:
    """
    Return the xy cross-sectional area of the simulation cell in Angstrom^2.
    Works for orthorhombic and triclinic boxes (non-zero xy xz yz tilts).
    |a x b| gives the area of the parallelogram base regardless of tilt.
    """
    cell = atoms.get_cell()
    a = cell[0]   # first lattice vector
    b = cell[1]   # second lattice vector
    # Full 3D cross product, take magnitude — correct for any tilt
    cross = np.cross(a, b)
    return float(np.linalg.norm(cross))


def get_lammps_types(atoms) -> np.ndarray:
    """
    Return the raw LAMMPS integer type column (1-indexed) for every atom.
    read_lammps_dump_frames() always injects these into atoms.arrays['type']
    directly from the dump text, bypassing ASE's atomic-number remapping.
    """
    if "type" in atoms.arrays:
        return np.array(atoms.arrays["type"], dtype=int)
    # Should not happen after read_lammps_dump_frames — but guard anyway.
    raise KeyError(
        "atoms.arrays['type'] not found. Was this frame read by "
        "read_lammps_dump_frames()? Available keys: "
        f"{list(atoms.arrays.keys())}"
    )


_type_diagnostic_done = False   # print once per run, not once per frame


def _print_type_diagnostic(atoms, adsorbate_types: list, substrate_types: list) -> None:
    """
    On the first frame, report what type integers ASE actually stored and
    check they overlap with the configured adsorbate/substrate type lists.
    Catches the common failure mode where ASE's "numbers" array contains
    atomic numbers (Al=13) rather than raw LAMMPS types (Al-type=4).
    """
    global _type_diagnostic_done
    if _type_diagnostic_done:
        return
    _type_diagnostic_done = True

    raw = get_lammps_types(atoms)
    unique = sorted(int(x) for x in np.unique(raw))
    all_expected = sorted(set(adsorbate_types) | set(substrate_types))
    missing = [t for t in all_expected if t not in unique]
    unexpected = [t for t in unique if t not in all_expected]

    src = "atoms.arrays['type']" if "type" in atoms.arrays else "atoms.arrays['numbers'] (fallback)"
    print(f"\n  --- Type mapping diagnostic (frame 0) ---")
    print(f"    Source array          : {src}")
    print(f"    Unique types in dump  : {unique}")
    print(f"    Expected (ads+sub)    : {all_expected}")
    if missing:
        print(f"    MISSING from dump     : {missing}  <- these will never match; check --adsorbate-types / --substrate-types")
    if unexpected:
        print(f"    Unexpected in dump    : {unexpected}  <- present but not in any type list")
    if not missing:
        print(f"    OK: all expected types are present in the dump.")
    print(f"  " + "-" * 40 + "\n")


# ---------------------------------------------------------------------------
# Surface z-reference detection
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class SurfaceZResult:
    """Diagnostics for one frame's auto-detected surface reference."""
    z_surface:           float  # mean z of the detected surface layer — used as the band's lower edge
    z_top:                float  # robust top-of-slab z (see detect_surface_z), NOT necessarily sub_z.max()
    z_top_absolute:       float  # the literal highest substrate atom z, before outlier rejection
    z_std:                 float  # std dev of z within the detected layer (roughness indicator)
    n_layer_atoms:         int    # how many substrate atoms fell inside slab_thickness of z_top
    n_substrate_total:     int    # total substrate atoms in the frame (sanity check on --substrate-types)
    n_stray_atoms:         int    # substrate atoms above z_top, excluded as likely outliers
    bulk_density:          float  # densest scan_bin_width slice found (atoms per bin) — the reference used
    density_threshold:     float  # actual atom-count threshold applied this frame (bulk_density * fraction)
    z_bin_edges:            object  # np.ndarray of histogram bin edges, for optional inspection/plotting
    z_bin_counts:            object  # np.ndarray of histogram counts, matching z_bin_edges


def detect_surface_z(positions: np.ndarray, types: np.ndarray,
                     substrate_types: list[int],
                     slab_thickness: float,
                     density_fraction: float = 0.3,
                     min_layer_atoms: int = 2,
                     scan_bin_width: float = 1.0) -> SurfaceZResult:
    """
    Find the z-coordinate of the topmost substrate layer, robust to a small
    number of stray substrate atoms (e.g. a reaction byproduct like H2O that
    still carries a substrate atom type but has desorbed and drifted away
    from the slab) without misclassifying genuine surface roughness as stray.

    Rather than a fixed absolute atom-count threshold (which has to be
    hand-tuned per system and breaks if slab thickness/area changes), this
    builds a z-density histogram of substrate atoms across the whole frame,
    takes the densest slice as a proxy for "bulk" density, and scans down
    from the absolute top for the first slice whose count reaches
    density_fraction * bulk_density (with min_layer_atoms as an absolute
    floor so a single stray atom still can't satisfy it). That slice's upper
    edge becomes z_top; substrate atoms above it are reported as strays.

    z_surface is then the mean z of substrate atoms within slab_thickness of
    this robust z_top.
    """
    sub_mask = np.isin(types, substrate_types)
    n_sub_total = int(sub_mask.sum())
    if n_sub_total == 0:
        raise ValueError("No substrate atoms found — check --substrate-types")

    sub_z = positions[sub_mask, 2]
    z_top_absolute = float(sub_z.max())
    z_bottom = float(sub_z.min())

    # Build a density profile across the whole substrate z-range. The
    # densest slice stands in for "bulk" density — this makes the surface
    # threshold self-scale to this specific slab (size, thickness, atom
    # count) instead of requiring a fixed absolute atom count to be
    # re-tuned by hand for every system.
    n_bins = max(1, int(np.ceil((z_top_absolute - z_bottom) / scan_bin_width)))
    edges = z_top_absolute - scan_bin_width * np.arange(n_bins + 1)   # top-down edges
    edges = edges[::-1]                                                # ascending for np.histogram
    counts, edges = np.histogram(sub_z, bins=edges)

    bulk_density = float(counts.max()) if counts.size else 0.0
    density_threshold = max(min_layer_atoms, density_fraction * bulk_density)

    # Scan top-down (counts/edges are ascending, so iterate in reverse)
    # for the first slice that meets the density threshold.
    z_top = z_top_absolute
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] >= density_threshold:
            z_top = float(edges[i + 1])   # upper edge of this slice
            break

    n_stray_atoms = int((sub_z > z_top).sum())

    # Surface layer = atoms within slab_thickness of the (robust) top
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
    chosen z_top marked. One row per scan_bin_width slice — lets you see
    exactly where the surface was placed relative to the real density
    profile, instead of trusting the auto-detection blindly.
    """
    lines = []
    max_count = int(counts.max()) if counts.size else 1
    for i in range(len(counts) - 1, -1, -1):
        lo, hi, c = edges[i], edges[i + 1], counts[i]
        bar = "#" * max(0, round(bar_width * c / max_count)) if max_count > 0 else ""
        marker = "  <- z_top" if abs(hi - z_top) < 1e-9 else ""
        lines.append(f"      [{lo:8.2f}, {hi:8.2f}) {c:5d}  {bar}{marker}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-frame coverage calculation
# ---------------------------------------------------------------------------

def compute_coverage(
    frame_index: int,
    atoms,
    adsorbate_types: list[int],
    substrate_types: list[int],
    surface_z_ref: float | str,
    slab_thickness: float,
    band_width: float,
    band_depth: float = 0.0,
    density_fraction: float = 0.3,
    min_layer_atoms: int = 2,
    scan_bin_width: float = 1.0,
) -> dict:
    """
    Returns a dict of coverage metrics for one frame.
    """
    positions = atoms.get_positions()                    # (N, 3)
    types = get_lammps_types(atoms)

    cell_area = get_cell_area(atoms)

    # Determine surface z reference
    if surface_z_ref == "auto":
        sz = detect_surface_z(positions, types, substrate_types, slab_thickness,
                               density_fraction=density_fraction,
                               min_layer_atoms=min_layer_atoms,
                               scan_bin_width=scan_bin_width)
        z_surf         = sz.z_surface
        z_top          = sz.z_top
        z_top_absolute = sz.z_top_absolute
        z_surf_std     = sz.z_std
        n_layer_atoms  = sz.n_layer_atoms
        n_stray_atoms  = sz.n_stray_atoms
    else:
        # Fixed reference: nothing was detected, so the diagnostic fields
        # below are reported as N/A-equivalents (0) rather than left undefined,
        # to keep the CSV schema identical between auto and fixed modes.
        z_surf         = float(surface_z_ref)
        z_top          = z_surf
        z_top_absolute = z_surf
        z_surf_std     = 0.0
        n_layer_atoms  = 0
        n_stray_atoms  = 0

    z_lo = z_surf - band_depth   # extends below surface to catch incorporated atoms
    z_hi = z_surf + band_width

    # Adsorbate atoms in the full band (z_lo to z_hi)
    ads_mask  = np.isin(types, adsorbate_types)
    band_mask = (positions[:, 2] >= z_lo) & (positions[:, 2] <= z_hi)
    surface_ads_mask = ads_mask & band_mask

    # Split into incorporated (below z_surface, i.e. reacted into slab)
    # and surface/gas-adjacent (above z_surface)
    above_mask = ads_mask & (positions[:, 2] >  z_surf) & (positions[:, 2] <= z_hi)
    below_mask = ads_mask & (positions[:, 2] <= z_surf) & (positions[:, 2] >= z_lo)

    n_ads_total        = int(ads_mask.sum())
    n_ads_surface      = int(surface_ads_mask.sum())   # full band
    n_ads_above        = int(above_mask.sum())          # above z_surface (physisorbed / approaching)
    n_ads_incorporated = int(below_mask.sum())          # below z_surface (reacted into slab)

    al_type = adsorbate_types[0]   # Al is type 4, first in default list
    al_mask = (types == al_type)
    n_al_surface      = int((al_mask & band_mask).sum())
    n_al_above        = int((al_mask & above_mask).sum())
    n_al_incorporated = int((al_mask & below_mask).sum())

    # Areal density — use Al above z_surface for the standard ALD metric
    # (incorporated Al counts toward total coverage but is already "done")
    areal_density_all = n_ads_surface / cell_area if cell_area > 0 else 0.0
    areal_density_al  = n_al_surface  / cell_area if cell_area > 0 else 0.0
    areal_density_al_above = n_al_above / cell_area if cell_area > 0 else 0.0
    areal_density_al_inc   = n_al_incorporated / cell_area if cell_area > 0 else 0.0

    return {
        "frame":                  frame_index,
        "surface_z_mode":         "auto" if surface_z_ref == "auto" else "fixed",
        "z_surface":              round(z_surf, 3),
        "z_top":                  round(z_top, 3),
        "z_top_absolute":         round(z_top_absolute, 3),
        "n_stray_substrate_atoms": n_stray_atoms,
        "z_surface_std":          round(z_surf_std, 3),
        "n_surface_layer_atoms":  n_layer_atoms,
        "z_band_lo":              round(z_lo, 3),
        "z_band_hi":              round(z_hi, 3),
        "cell_area_A2":           round(cell_area, 3),
        "n_adsorbate_total":      n_ads_total,
        "n_adsorbate_surface":    n_ads_surface,       # full band (above + incorporated)
        "n_adsorbate_above":      n_ads_above,          # above z_surface only
        "n_adsorbate_incorporated": n_ads_incorporated, # below z_surface (reacted into slab)
        "n_Al_surface":           n_al_surface,
        "n_Al_above":             n_al_above,
        "n_Al_incorporated":      n_al_incorporated,
        "areal_density_all":      round(areal_density_all, 6),      # atoms/Å² (full band)
        "areal_density_Al":       round(areal_density_al,  6),      # atoms/Å² (Al, full band)
        "areal_density_nm2":      round(areal_density_al * 100, 4), # Al atoms/nm² (full band)
        "areal_density_Al_above": round(areal_density_al_above * 100, 4),  # Al/nm² above surface
        "areal_density_Al_inc":   round(areal_density_al_inc * 100, 4),    # Al/nm² incorporated
    }


# ---------------------------------------------------------------------------
# Z-density profile (histogram across all frames)
# ---------------------------------------------------------------------------

def compute_z_profiles(
    dump_path: str,
    adsorbate_types: list[int],
    substrate_types: list[int],
    stride: int,
    z_bin_width: float = Z_BIN_WIDTH,
) -> pd.DataFrame:
    """
    Build a 2D array: (frame x z_bin) -> atom count.
    Returns a DataFrame indexed by frame, columns = z bin centres.
    """
    print("  Computing z-density profiles...")
    rows = {}

    for frame_index, atoms in read_lammps_dump_frames(dump_path, stride):
        positions = atoms.get_positions()
        types = get_lammps_types(atoms)

        ads_mask = np.isin(types, adsorbate_types)
        z_vals   = positions[ads_mask, 2]

        if len(z_vals) == 0:
            continue

        z_min = 0.0
        z_max = atoms.get_cell()[2, 2]   # box z-length
        bins  = np.arange(z_min, z_max + z_bin_width, z_bin_width)
        counts, edges = np.histogram(z_vals, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])

        rows[frame_index] = dict(zip(np.round(centres, 2), counts))

    df = pd.DataFrame(rows).T.fillna(0).astype(int)
    df.index.name = "frame"
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Surface-detection diagnostics (printed once, on the first frame)
# ---------------------------------------------------------------------------

def _print_surface_detection_summary(
    rec: dict,
    sz,
    surface_z_ref: float | str,
    slab_thickness: float,
    band_width: float,
    density_fraction: float,
    min_layer_atoms: int,
    scan_bin_width: float,
    atoms=None,
) -> None:
    """
    Print an explicit, human-checkable summary of how z_surface was determined
    on the first frame, so a bad --slab-thickness or density setting is
    obvious before waiting on the rest of the trajectory. sz is the full
    SurfaceZResult (with histogram) when in auto mode, or None in fixed mode.
    """
    print("\n  --- Surface detection (frame {}) ---".format(rec["frame"]))
    if atoms is not None:
        cell_z = float(atoms.get_cell()[2, 2])
        pbc = tuple(bool(p) for p in atoms.get_pbc())
        print(f"    Box z-length (cell[2,2])         : {cell_z:.3f} Å")
        print(f"    PBC flags (x, y, z)              : {pbc}")
        if pbc[2]:
            print(f"    NOTE: z is periodic. If the histogram below shows atoms split into separate "
                  f"dense regions with real gaps between them, and you expect one continuous slab, "
                  f"check whether the dump stores wrapped (x y z) or unwrapped (xu yu zu) coordinates — "
                  f"a continuous slab can appear split across the z boundary in wrapped coordinates.")
    if rec["surface_z_mode"] == "auto" and sz is not None:
        print(f"    Mode              : auto")
        print(f"    Literal highest substrate atom  : {sz.z_top_absolute:.3f} Å")
        print(f"    Bulk density (densest {scan_bin_width} Å slice) : {sz.bulk_density:.0f} atoms")
        print(f"    Density threshold applied        : {sz.density_threshold:.1f} atoms "
              f"(= max({min_layer_atoms} floor, {density_fraction} x bulk_density))")
        if sz.n_stray_atoms > 0:
            print(f"    -> {sz.n_stray_atoms} substrate atom(s) above the robust surface "
                  f"were treated as strays and excluded (see z_top vs z_top_absolute)")
        print(f"    Robust z_top (used as reference) : {sz.z_top:.3f} Å")
        print(f"    slab_thickness used              : {slab_thickness} Å")
        print(f"    -> layer scanned                 : [{sz.z_top - slab_thickness:.3f}, {sz.z_top:.3f}] Å")
        print(f"    Substrate atoms in that layer     : {rec['n_surface_layer_atoms']}")
        print(f"    z_surface (mean of layer)         : {rec['z_surface']:.3f} Å")
        print(f"    z_surface roughness (std)         : {rec['z_surface_std']:.3f} Å")

        print(f"\n    z-density histogram (substrate atoms, {scan_bin_width} Å bins, top to bottom):")
        print(format_z_histogram(sz.z_bin_edges, sz.z_bin_counts, sz.z_top))
        print(f"    Look at where the count actually drops off above — if z_top doesn't land where "
              f"you'd visually call the surface edge, adjust --surface-density-fraction or "
              f"--min-layer-atoms and rerun; frame 0 is cheap to check.")

        # Sanity checks — heuristics, not hard errors, since what counts as
        # "too few atoms" or "too rough" depends on your system size.
        if rec["n_surface_layer_atoms"] < 10:
            print(f"\n    WARNING: only {rec['n_surface_layer_atoms']} substrate atoms fell inside "
                  f"slab_thickness={slab_thickness} Å — consider increasing --slab-thickness.")
        if rec["z_surface_std"] > slab_thickness / 2:
            print(f"    WARNING: z_surface_std ({rec['z_surface_std']:.3f} Å) is more than half of "
                  f"slab_thickness ({slab_thickness} Å) — the 'surface layer' may span more than one "
                  f"physical layer, or roughness makes a single mean z a shaky reference.")
    else:
        print(f"    Mode              : fixed")
        print(f"    z_surface (user-specified)       : {rec['z_surface']:.3f} Å")
        print(f"    (auto-detection diagnostics n/a in fixed mode)")

    print(f"\n    Surface band scanned for adsorbate : [{rec['z_band_lo']:.3f}, {rec['z_band_hi']:.3f}] Å "
          f"(z_surface + band_width={band_width} Å)")
    print("  " + "-" * 40 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    dump_path: str,
    output_path: str,
    adsorbate_types: list[int],
    substrate_types: list[int],
    surface_z_ref: float | str,
    slab_thickness: float,
    band_width: float,
    band_depth: float,
    stride: int,
    density_fraction: float = 0.3,
    min_layer_atoms: int = 2,
    scan_bin_width: float = 1.0,
):
    print(f"Loading trajectory: {dump_path}")
    print(f"  Adsorbate types : {adsorbate_types}")
    print(f"  Substrate types : {substrate_types}")
    print(f"  Surface z ref   : {surface_z_ref}")
    print(f"  Band width      : {band_width} Å  (above z_surface)")
    print(f"  Band depth      : {band_depth} Å  (below z_surface, captures incorporated atoms)")
    print(f"  Stride          : {stride}")
    if surface_z_ref == "auto":
        print(f"  Density fraction: {density_fraction}  (surface accepted where density >= this x bulk density)")
        print(f"  Min layer atoms : {min_layer_atoms}  (absolute floor under the density threshold)")
        print(f"  Scan bin width  : {scan_bin_width} Å  (slice size used when scanning down for z_top)")

    records = []
    frame_count = 0

    for frame_index, atoms in read_lammps_dump_frames(dump_path, stride):
        if frame_count % 50 == 0:
            print(f"  Frame {frame_index}...")

        rec = compute_coverage(
            frame_index, atoms,
            adsorbate_types, substrate_types,
            surface_z_ref, slab_thickness, band_width,
            band_depth=band_depth,
            density_fraction=density_fraction,
            min_layer_atoms=min_layer_atoms,
            scan_bin_width=scan_bin_width,
        )
        records.append(rec)

        if frame_count == 0:
            _print_type_diagnostic(atoms, adsorbate_types, substrate_types)
            sz = None
            if surface_z_ref == "auto":
                positions = atoms.get_positions()
                types = get_lammps_types(atoms)
                sz = detect_surface_z(positions, types, substrate_types, slab_thickness,
                                       density_fraction=density_fraction,
                                       min_layer_atoms=min_layer_atoms,
                                       scan_bin_width=scan_bin_width)
            _print_surface_detection_summary(
                rec, sz, surface_z_ref, slab_thickness, band_width,
                density_fraction, min_layer_atoms, scan_bin_width,
                atoms=atoms,
            )

        frame_count += 1

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\n  Coverage data -> {output_path}")

    # Z profiles
    z_profile_path = Path(output_path).with_suffix("").with_suffix(".z_profile.csv")
    z_df = compute_z_profiles(dump_path, adsorbate_types, substrate_types, stride)
    z_df.to_csv(z_profile_path)
    print(f"  Z profiles     -> {z_profile_path}")

    # Summary
    print(f"\nSummary statistics:")
    print(f"  Frames analysed : {len(df)}")
    if not df.empty:
        print(f"  Al areal density (final frame): {df['areal_density_Al'].iloc[-1]:.4f} atoms/Å²"
              f"  = {df['areal_density_nm2'].iloc[-1]:.2f} atoms/nm²")
        print(f"  Peak surface adsorbate count  : {df['n_adsorbate_surface'].max()}")
        print(f"  Mean z_surface                : {df['z_surface'].mean():.2f} ± "
              f"{df['z_surface'].std():.2f} Å")

    return df, z_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_coverage(df: pd.DataFrame, z_df: pd.DataFrame, output_stem: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    # --- Coverage vs time ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(df["frame"], df["n_adsorbate_surface"], color="#3B6D11", linewidth=1.2, label="All adsorbate atoms")
    axes[0].plot(df["frame"], df["n_Al_surface"],        color="#185FA5", linewidth=1.2, label="Al atoms only")
    axes[0].set_ylabel("Atoms in surface band")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Surface coverage over trajectory")

    axes[1].plot(df["frame"], df["areal_density_nm2"], color="#BA7517", linewidth=1.2)
    axes[1].set_ylabel("Al areal density (atoms/nm²)")
    axes[1].set_xlabel("Frame")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    cov_png = f"{output_stem}_coverage.png"
    fig.savefig(cov_png, dpi=150)
    print(f"  Plot saved -> {cov_png}")
    plt.close(fig)

    # --- Z-density heatmap ---
    if not z_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        z_cols = z_df.columns.astype(float)
        im = ax.pcolormesh(
            z_cols,
            z_df.index,
            z_df.values,
            cmap="YlOrRd",
            norm=mcolors.PowerNorm(gamma=0.4),
            shading="auto",
        )
        fig.colorbar(im, ax=ax, label="Atom count per bin")
        ax.set_xlabel("z (Å)")
        ax.set_ylabel("Frame")
        ax.set_title("Adsorbate z-density over time")
        fig.tight_layout()
        z_png = f"{output_stem}_z_profile.png"
        fig.savefig(z_png, dpi=150)
        print(f"  Plot saved -> {z_png}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface coverage analysis for LAMMPS TMA/SiO2 trajectories")

    parser.add_argument("--dump",    required=True,  help="LAMMPS dump file")
    parser.add_argument("--output",  required=True,  help="Output CSV for coverage data")

    parser.add_argument("--adsorbate-types", type=int, nargs="+",
                        default=DEFAULT_ADSORBATE_TYPES,
                        help="LAMMPS atom types for adsorbate atoms (default: 1 2 3)")
    parser.add_argument("--substrate-types", type=int, nargs="+",
                        default=DEFAULT_SUBSTRATE_TYPES,
                        help="LAMMPS atom types for substrate atoms (default: 4 5)")

    parser.add_argument("--surface-z-ref", default=DEFAULT_SURFACE_Z_REF,
                        help="'auto' or a fixed float (Å). auto detects from topmost substrate atom.")
    parser.add_argument("--slab-thickness", type=float, default=DEFAULT_SLAB_THICKNESS,
                        help="Depth below topmost substrate atom counted as surface layer (default: 3.0 Å)")
    parser.add_argument("--band-width",     type=float, default=DEFAULT_BAND_WIDTH,
                        help="Height above z_surface to count as 'on surface' (default: 8.0 Å)")
    parser.add_argument("--band-depth",     type=float, default=DEFAULT_BAND_DEPTH,
                        help="Depth below z_surface to count as 'on surface' — captures adsorbate atoms "
                             "that have reacted and incorporated into the slab (default: 0.0 Å, above-surface only). "
                             "Set to ~12 Å for TMA/SiO2 to catch all incorporated Al.")
    parser.add_argument("--min-layer-atoms", type=int, default=2,
                        help="Absolute floor under the density threshold when auto-detecting z_top — "
                             "guards against a single stray atom satisfying the threshold on a very "
                             "sparse slab (default: 2)")
    parser.add_argument("--surface-density-fraction", type=float, default=0.3, dest="density_fraction",
                        help="When auto-detecting z_top, a z-slice is trusted as the real slab surface "
                             "once its atom count reaches this fraction of the slab's own bulk (densest-slice) "
                             "density. Self-scales to slab size/thickness, unlike a fixed atom count. "
                             "Lower = trust sparser/rougher slices as the surface; higher = require denser "
                             "slices, pushing z_top further down (default: 0.3)")
    parser.add_argument("--scan-bin-width", type=float, default=1.0,
                        help="Slice thickness (Å) used when building the density profile and scanning down "
                             "from the absolute top to find z_top (default: 1.0 Å)")

    parser.add_argument("--stride",  type=int, default=1,
                        help="Analyse every Nth frame (default: 1)")
    parser.add_argument("--plot",    action="store_true",
                        help="Generate matplotlib plots alongside CSV output")

    args = parser.parse_args()

    # Allow 'auto' or a numeric string for surface-z-ref
    try:
        z_ref = float(args.surface_z_ref)
    except ValueError:
        z_ref = "auto"

    df, z_df = run(
        dump_path        = args.dump,
        output_path      = args.output,
        adsorbate_types  = args.adsorbate_types,
        substrate_types  = args.substrate_types,
        surface_z_ref    = z_ref,
        slab_thickness   = args.slab_thickness,
        band_width       = args.band_width,
        band_depth       = args.band_depth,
        stride           = args.stride,
        density_fraction = args.density_fraction,
        min_layer_atoms  = args.min_layer_atoms,
        scan_bin_width   = args.scan_bin_width,
    )

    if args.plot:
        stem = str(Path(args.output).with_suffix(""))
        plot_coverage(df, z_df, stem)
