"""
deposition_analysis.py
-----------------------
Master analysis script for TMA/SiO2 ALD deposition trajectories. Combines:

  - Species tracking (species_tracker.py): identifies molecular fragments
    by connectivity + composition (TMA, DMA*, CH4 byproduct, etc.)
  - Surface coverage (surface_coverage.py): robust, density-based
    z_surface detection, validated against real slab data.
  - Adsorption-state classification (new): chemisorbed vs physisorbed vs
    incorporated vs unbound, using TWO cluster-analysis passes per frame
    (a normal bonding cutoff and a looser contact cutoff) rather than a
    z-position-only heuristic. See adsorption_states.py for definitions.

Uses OVITO for trajectory reading and cluster analysis (bonding
connectivity), and the shared lammps_common.detect_surface_z for the
surface reference — both fed from the same OVITO DataCollection arrays,
so the trajectory is only read once.

Usage:
    python deposition_analysis.py --dump dump1.lammpstrj --output analysis.csv
    python deposition_analysis.py --dump dump1.lammpstrj --output analysis.csv \\
        --chem-cutoff 2.2 --contact-cutoff 4.0 --stride 10

Requires: ovito, numpy, pandas
Install:  pip install ovito numpy pandas
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from ovito.io import import_file
from ovito.modifiers import ClusterAnalysisModifier

from lammps_common import detect_surface_z, format_z_histogram
from adsorption_states import classify_fragments, FragmentState
from al_bonding_analysis import analyse_al_atoms, AlAtomRecord


# ---------------------------------------------------------------------------
# Configuration — edit to match your LAMMPS atom types
# ---------------------------------------------------------------------------

ATOM_TYPE_MAP = {
    1: "H",    # substrate hydroxyl H
    2: "O",    # substrate oxygen
    3: "Si",   # substrate silicon
    4: "Al",   # TMA aluminium
    5: "C",    # methyl carbon
    6: "H",    # methyl hydrogen (TMA-side)
}

SUBSTRATE_TYPES = {1, 2, 3}
ADSORBATE_TYPES = {4, 5, 6}
AL_TYPE = 4       # LAMMPS type for the metal atom whose ligand state we track
CARBON_TYPE = 5   # LAMMPS type for methyl carbon (ligand-counting target)

# Normal (covalent) bonding cutoff — same as species_tracker.py's DEFAULT_CUTOFF.
DEFAULT_CHEM_CUTOFF = 2.2

# Looser "contact" cutoff for physisorption detection (van der Waals range).
# Must be larger than DEFAULT_CHEM_CUTOFF or physisorption can never be
# distinguished from chemisorption.
DEFAULT_CONTACT_CUTOFF = 4.0

# NOTE on why Al ligand/binding state uses a SEPARATE, per-atom analysis
# (al_bonding_analysis.analyse_al_atoms) rather than the cluster-based
# species/state table below: at moderate-to-high coverage, individual
# chemisorbed fragments bond to each other through the substrate (real
# Al-O-Al bridging), so cluster analysis correctly merges them into one
# large connected network — which makes a whole-cluster composition
# string like "Al36C83H150" meaningless as a per-atom reaction-progress
# indicator. The species/state table below remains useful for tracking
# genuinely separate small fragments (gas-phase byproducts: CH4, H2, H*,
# C2H6, and intact/dimerized precursor) — but for "how many ligands does
# THIS Al atom still carry, and is it directly bonded to the slab", see
# the al_atoms outputs instead, which check distances from each Al atom
# individually and are immune to network percolation.

# Surface-detection defaults (see lammps_common.detect_surface_z)
DEFAULT_SLAB_THICKNESS = 3.0
DEFAULT_DENSITY_FRACTION = 0.3
DEFAULT_MIN_LAYER_ATOMS = 2
DEFAULT_SCAN_BIN_WIDTH = 1.0

# Composition fingerprints -> human labels. Extend this as new byproducts
# turn up in your KNOWN_SPECIES-miss warnings (see label_species()).
KNOWN_SPECIES = {
    frozenset([("Al", 1), ("C", 3), ("H", 9)]):  "TMA",
    frozenset([("Al", 1), ("C", 2), ("H", 6)]):  "DMA*",
    frozenset([("Al", 1), ("C", 1), ("H", 3)]):  "MMA*",
    frozenset([("Al", 1)]):                        "Al*",
    frozenset([("C", 1), ("H", 3)]):              "CH3",
    frozenset([("C", 1), ("H", 4)]):              "CH4",
    frozenset([("H", 2)]):                          "H2",
    frozenset([("H", 1)]):                          "H*",       # atomic H radical
    frozenset([("Al", 2), ("C", 4), ("H", 12)]): "TMA-dimer",
    frozenset([("Al", 1), ("C", 2), ("H", 6), ("O", 1)]): "DMA-O*",
    frozenset([("C", 2), ("H", 6)]):              "C2H6",       # ethane byproduct
}


# ---------------------------------------------------------------------------
# Species labelling (from species_tracker.py)
# ---------------------------------------------------------------------------

def composition_fingerprint(atom_types: list) -> frozenset:
    element_counts = Counter(ATOM_TYPE_MAP.get(t, f"T{t}") for t in atom_types)
    return frozenset(element_counts.items())


def label_species(fingerprint: frozenset) -> str:
    if fingerprint in KNOWN_SPECIES:
        return KNOWN_SPECIES[fingerprint]
    parts = sorted(fingerprint, key=lambda x: x[0])
    return "".join(f"{el}{n}" if n > 1 else el for el, n in parts) or "empty"


def fragment_anchor_key(ads_types: list, ads_ids: list) -> str:
    """
    A persistent identity for one fragment, used to tell "same physical
    molecule, later frame" from "a new one" during bulk analysis — the
    per-frame cluster analysis has no memory between frames on its own.

    Anchored to whichever atom(s) are chemically least likely to swap
    between fragments over the fragment's lifetime:
      - Al-containing fragment  -> anchor = the Al atom id(s)
      - Carbon-containing (no Al) -> anchor = the carbon atom id(s)
      - Pure-H fragment (H2, H*) -> anchor = the H atom id(s) themselves
    This is a real approximation, not a guarantee — an H2 molecule could
    in principle exchange one of its atoms for a different H over time,
    which would be invisible to this scheme. Worth keeping in mind for
    H2/H* counts specifically; the Al- and C-anchored counts are on much
    firmer chemical ground (a carbon essentially never leaves a fragment
    once bonded, short of a bond-breaking event this cutoff regime
    wouldn't represent well anyway).
    """
    al_ids = sorted(i for t, i in zip(ads_types, ads_ids) if ATOM_TYPE_MAP.get(t) == "Al")
    if al_ids:
        return "Al:" + ",".join(map(str, al_ids))
    c_ids = sorted(i for t, i in zip(ads_types, ads_ids) if ATOM_TYPE_MAP.get(t) == "C")
    if c_ids:
        return "C:" + ",".join(map(str, c_ids))
    h_ids = sorted(i for t, i in zip(ads_types, ads_ids) if ATOM_TYPE_MAP.get(t) == "H")
    return "H:" + ",".join(map(str, h_ids))


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def analyse_frame(
    frame_index: int,
    positions: np.ndarray,
    types: np.ndarray,
    ids: np.ndarray,
    chem_cluster_ids: np.ndarray,
    contact_cluster_ids: np.ndarray,
    z_surface: float,
) -> list[dict]:
    """
    Classify every adsorbate fragment in one frame and return one record
    per fragment: species label, adsorption state, position, composition,
    and an anchor_key identifying it across frames (see fragment_anchor_key).
    """
    frag_states = classify_fragments(
        types, positions, chem_cluster_ids, contact_cluster_ids,
        substrate_types=SUBSTRATE_TYPES, z_surface=z_surface,
    )

    records = []
    for fs in frag_states:
        frag_mask = (chem_cluster_ids == fs.cluster_id_chem) & np.isin(types, list(ADSORBATE_TYPES))
        ads_types = types[frag_mask].tolist()
        ads_ids = ids[frag_mask].tolist()
        fp = composition_fingerprint(ads_types)
        label = label_species(fp)
        anchor_key = fragment_anchor_key(ads_types, ads_ids)

        records.append({
            "frame":         frame_index,
            "species":       label,
            "state":         fs.state,      # unbound | physisorbed | chemisorbed | incorporated
            "anchor_key":    anchor_key,
            "n_ads_atoms":   fs.n_ads_atoms,
            "n_sub_bonded":  fs.n_sub_atoms_bonded,
            "z_mean":        round(fs.z_mean, 3),
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
    chem_cutoff: float = DEFAULT_CHEM_CUTOFF,
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF,
    al_c_cutoff: float | None = None,
    scan_cutoffs: list[float] | None = None,
    stride: int = 1,
    slab_thickness: float = DEFAULT_SLAB_THICKNESS,
    density_fraction: float = DEFAULT_DENSITY_FRACTION,
    min_layer_atoms: int = DEFAULT_MIN_LAYER_ATOMS,
    scan_bin_width: float = DEFAULT_SCAN_BIN_WIDTH,
    site_by_id: dict | None = None,
):
    if contact_cutoff <= chem_cutoff:
        raise ValueError(
            f"--contact-cutoff ({contact_cutoff}) must be larger than "
            f"--chem-cutoff ({chem_cutoff}), or physisorption can never be "
            f"distinguished from chemisorption."
        )
    if al_c_cutoff is None:
        al_c_cutoff = chem_cutoff
    if scan_cutoffs is None:
        scan_cutoffs = sorted(set([2.0, chem_cutoff, 3.0, contact_cutoff]))

    print(f"Loading trajectory: {dump_path}")
    pipeline = import_file(dump_path, multiple_frames=True)
    chem_modifier = ClusterAnalysisModifier(cutoff=chem_cutoff, sort_by_size=False, compute_com=False)
    pipeline.modifiers.append(chem_modifier)

    n_frames = pipeline.source.num_frames
    frame_indices = list(range(0, n_frames, stride))
    print(f"  {n_frames} frames found, stride={stride}, effective frames={len(frame_indices)}")
    print(f"  Chemical (bonding) cutoff : {chem_cutoff} Å")
    print(f"  Contact (physisorption) cutoff : {contact_cutoff} Å")
    print(f"  Al-C ligand cutoff : {al_c_cutoff} Å")
    print(f"  Substrate-neighbor scan cutoffs : {scan_cutoffs} Å")

    all_records = []
    all_al_records = []

    for i, frame in enumerate(frame_indices):
        if i % 50 == 0:
            print(f"  Processing frame {frame}/{n_frames} ({100*frame//n_frames}%)...")

        # Pass 1: chemical (bonding) cutoff clusters
        chem_modifier.cutoff = chem_cutoff
        data_chem = pipeline.compute(frame)
        positions = np.array(data_chem.particles["Position"][:])
        types = np.array(data_chem.particles["Particle Type"][:])
        ids = np.array(data_chem.particles["Particle Identifier"][:])
        chem_cluster_ids = np.array(data_chem.particles["Cluster"][:])

        # Pass 2: contact (physisorption) cutoff clusters — same frame,
        # different cutoff. Re-running compute() with the modifier's
        # cutoff changed re-executes the pipeline at that cutoff.
        chem_modifier.cutoff = contact_cutoff
        data_contact = pipeline.compute(frame)
        contact_cluster_ids = np.array(data_contact.particles["Cluster"][:])
        chem_modifier.cutoff = chem_cutoff   # restore for next frame's pass 1

        # Robust surface detection (shared, validated logic)
        sz = detect_surface_z(
            positions, types, list(SUBSTRATE_TYPES), slab_thickness,
            density_fraction=density_fraction,
            min_layer_atoms=min_layer_atoms,
            scan_bin_width=scan_bin_width,
        )

        if i == 0:
            print(f"\n  --- Surface detection (frame {frame}) ---")
            print(f"    z_top_absolute : {sz.z_top_absolute:.3f} Å")
            print(f"    z_top (robust) : {sz.z_top:.3f} Å  ({sz.n_stray_atoms} stray atom(s) excluded)")
            print(f"    z_surface      : {sz.z_surface:.3f} Å  (std {sz.z_std:.3f} Å, "
                  f"{sz.n_layer_atoms} atoms)")
            print(format_z_histogram(sz.z_bin_edges, sz.z_bin_counts, sz.z_top))
            print("  " + "-" * 40 + "\n")

        records = analyse_frame(frame, positions, types, ids, chem_cluster_ids,
                                 contact_cluster_ids, sz.z_surface)
        for r in records:
            r["z_surface"] = round(sz.z_surface, 3)
        all_records.extend(records)

        # Per-Al-atom local ligand/binding analysis — direct distances,
        # NOT cluster connectivity, so it's immune to network percolation.
        al_recs = analyse_al_atoms(
            data_chem, frame, AL_TYPE, CARBON_TYPE, SUBSTRATE_TYPES,
            sz.z_surface, chem_cutoff, contact_cutoff,
            al_c_cutoff=al_c_cutoff, scan_cutoffs=scan_cutoffs,
            site_by_id=site_by_id,
        )
        for r in al_recs:
            row = {
                "frame": r.frame, "lammps_id": r.lammps_id,
                "x": round(r.x, 3), "y": round(r.y, 3), "z": round(r.z, 3),
                "z_surface": round(r.z_surface, 3), "z_rel": round(r.z_rel, 3),
                "n_C_ligands": r.n_C_ligands, "ligand_state": r.ligand_state,
                "nearest_substrate_dist": round(r.nearest_substrate_dist, 3) if r.nearest_substrate_dist == r.nearest_substrate_dist else None,
                "nearest_substrate_id": r.nearest_substrate_id,
                "binding_state": r.binding_state,
                "bound_site_type": r.bound_site_type,
            }
            for c, n in r.n_substrate_neighbors.items():
                row[f"n_sub_within_{c}"] = n
            all_al_records.append(row)

    df = pd.DataFrame(all_records)
    al_df = pd.DataFrame(all_al_records)

    # --- Per-frame state summary: counts and Al breakdown by state ---
    if not df.empty:
        state_counts = (
            df.groupby(["frame", "state"]).size().unstack(fill_value=0)
        )
        al_by_state = (
            df.groupby(["frame", "state"])["n_Al"].sum().unstack(fill_value=0)
        )
        al_by_state.columns = [f"n_Al_{c}" for c in al_by_state.columns]
        summary = state_counts.join(al_by_state, how="outer").fillna(0).astype(int)
    else:
        summary = pd.DataFrame()

    # --- Species breakdown (unbound/physisorbed only — "what's released") ---
    # NOTE: multi-Al entries here (e.g. TMA dimers reacting in the gas phase)
    # are still legitimate small connected fragments, not percolated network
    # artifacts, since unbound/physisorbed fragments by definition aren't
    # merged with the substrate. For per-Al ligand/binding state, use the
    # al_atoms outputs below instead — that's the authoritative source.
    released = df[df["state"].isin(["unbound", "physisorbed"])] if not df.empty else df
    if not released.empty:
        species_over_time = (
            released.groupby(["frame", "species"]).size()
                     .unstack(fill_value=0)
        )
    else:
        species_over_time = pd.DataFrame()

    # --- Per-Al-atom summary: ligand_state x binding_state counts per frame ---
    if not al_df.empty:
        al_state_counts = (
            al_df.groupby(["frame", "ligand_state", "binding_state"]).size()
                 .unstack(["ligand_state", "binding_state"], fill_value=0)
        )
        al_state_counts.columns = [f"{l}|{b}" for l, b in al_state_counts.columns]
    else:
        al_state_counts = pd.DataFrame()

    # --- Save outputs ---
    raw_out       = Path(output_path).with_suffix(".raw.csv")
    summary_out   = Path(output_path)
    species_out   = Path(output_path).with_name(Path(output_path).stem + "_species.csv")
    al_raw_out    = Path(output_path).with_name(Path(output_path).stem + "_al_atoms.raw.csv")
    al_states_out = Path(output_path).with_name(Path(output_path).stem + "_al_states.csv")

    df.to_csv(raw_out, index=False)
    if not summary.empty:
        summary.to_csv(summary_out)
    if not species_over_time.empty:
        species_out_written = True
        species_over_time.to_csv(species_out)
    else:
        species_out_written = False
    al_df.to_csv(al_raw_out, index=False)
    if not al_state_counts.empty:
        al_state_counts.to_csv(al_states_out)

    print(f"\nDone.")
    print(f"  Raw fragment data       -> {raw_out}")
    print(f"  Per-frame state summary -> {summary_out}")
    if species_out_written:
        print(f"  Released species        -> {species_out}")
    print(f"  Per-Al-atom raw data    -> {al_raw_out}")
    if not al_state_counts.empty:
        print(f"  Per-Al ligand/binding summary -> {al_states_out}")

    if not df.empty:
        last_frame = df["frame"].max()
        last = df[df["frame"] == last_frame]
        print(f"\nFinal-frame fragment breakdown (frame {last_frame}, cluster-based — "
              f"chemisorbed/incorporated entries here may show merged multi-Al network "
              f"compositions; see per-Al breakdown below for individual ligand counts):")
        for state in ["incorporated", "chemisorbed", "physisorbed", "unbound"]:
            sub = last[last["state"] == state]
            n_al = int(sub["n_Al"].sum())
            print(f"  {state:14s}: {len(sub):4d} fragments, {n_al:4d} Al atoms")

    if not al_df.empty:
        last_frame_al = al_df["frame"].max()
        last_al = al_df[al_df["frame"] == last_frame_al]
        binding_order = ["unbound", "physisorbed", "chemisorbed", "incorporated"]
        # Standard states first, then anything else actually present (e.g.
        # "unexpected(n)" from overlapping ligand-count geometry) so nothing
        # is silently dropped from the printed total.
        standard_states = ["TMA-like", "DMA-like", "MMA-like", "Al*-bare"]
        present_states = last_al["ligand_state"].unique().tolist()
        other_states = sorted(s for s in present_states if s not in standard_states)
        ligand_order = standard_states + other_states

        print(f"\nFinal-frame per-Al breakdown (frame {last_frame_al}, {len(last_al)} Al atoms total):")
        print(f"  {'':14s}  " + "  ".join(f"{s:>10s}" for s in binding_order))
        total_accounted = 0
        for ligand_state in ligand_order:
            counts = []
            for binding_state in binding_order:
                n = int(((last_al["ligand_state"] == ligand_state) & (last_al["binding_state"] == binding_state)).sum())
                counts.append(n)
            row_total = sum(counts)
            total_accounted += row_total
            if row_total > 0:
                print(f"  {ligand_state:14s}: " + "  ".join(f"{c:10d}" for c in counts))
        if other_states:
            print(f"\n  NOTE: {sum((last_al['ligand_state']==s).sum() for s in other_states)} Al atom(s) had "
                  f"unexpected ligand counts (see rows above) — usually means two Al centers are close "
                  f"enough that a neighboring Al's own methyl carbon falls within --al-c-cutoff of this "
                  f"one too. Worth checking those specific atoms in analysis_al_atoms.raw.csv "
                  f"(filter ligand_state.str.startswith('unexpected')) and cross-referencing against "
                  f"the visual snapshot.")
        if total_accounted != len(last_al):
            print(f"  WARNING: table sums to {total_accounted} but {len(last_al)} Al atoms were found "
                  f"this frame — investigate before trusting these counts.")

        n_dma_chem = int(((last_al["ligand_state"] == "DMA-like") &
                           (last_al["binding_state"] == "chemisorbed")).sum())
        print(f"\n  DMA-like + chemisorbed (lost 1 ligand, bonded on top, ready for next half-cycle) "
              f"= {n_dma_chem} Al atoms")

    return df, summary, species_over_time, al_df, al_state_counts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master TMA/SiO2 deposition analysis: species + surface coverage + adsorption states"
    )
    parser.add_argument("--dump",   required=True, help="Path to LAMMPS dump file")
    parser.add_argument("--output", required=True, help="Output CSV path (per-frame state summary)")
    parser.add_argument("--dump0", default=None,
                        help="Path to the pristine pre-deposition dump (dump0.lammpstrj). If given, "
                             "classifies surface O/Si sites (hydroxyl_O/bridging_O/dangling_O/"
                             "*_Si — see site_classification.py) and tags each chemisorbed/"
                             "incorporated Al with the site type it bonded to (bound_site_type "
                             "column). Optional — omit to skip site classification entirely.")
    parser.add_argument("--si-o-cutoff", type=float, default=2.0,
                        help="Si-O bonding cutoff for site classification, Å (default: 2.0 — "
                             "validate against this slab's own Si-O distance histogram before trusting)")
    parser.add_argument("--o-h-cutoff", type=float, default=1.2,
                        help="O-H bonding cutoff for site classification, Å (default: 1.2 — "
                             "validate against this slab's own O-H distance histogram before trusting)")
    parser.add_argument("--chem-cutoff", type=float, default=DEFAULT_CHEM_CUTOFF,
                        help=f"Covalent bonding cutoff, Å (default: {DEFAULT_CHEM_CUTOFF})")
    parser.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF,
                        help=f"Physisorption/contact cutoff, Å, must be > --chem-cutoff "
                             f"(default: {DEFAULT_CONTACT_CUTOFF})")
    parser.add_argument("--al-c-cutoff", type=float, default=None,
                        help="Distance defining an intact Al-C ligand bond, Å. "
                             "Defaults to --chem-cutoff if not given.")
    parser.add_argument("--scan-cutoffs", type=str, default=None,
                        help="Comma-separated list of distances (Å) to report Al-substrate "
                             "neighbor counts at, e.g. '2.0,2.2,3.0,4.0'. Defaults to "
                             "[2.0, chem_cutoff, 3.0, contact_cutoff].")
    parser.add_argument("--stride", type=int, default=1,
                        help="Analyse every Nth frame (default: 1)")
    parser.add_argument("--slab-thickness", type=float, default=DEFAULT_SLAB_THICKNESS,
                        help=f"Depth below z_top counted as surface layer, Å (default: {DEFAULT_SLAB_THICKNESS})")
    parser.add_argument("--surface-density-fraction", type=float, default=DEFAULT_DENSITY_FRACTION,
                        dest="density_fraction",
                        help=f"Density fraction for robust z_top detection (default: {DEFAULT_DENSITY_FRACTION})")
    parser.add_argument("--min-layer-atoms", type=int, default=DEFAULT_MIN_LAYER_ATOMS,
                        help=f"Absolute floor under the density threshold (default: {DEFAULT_MIN_LAYER_ATOMS})")
    parser.add_argument("--scan-bin-width", type=float, default=DEFAULT_SCAN_BIN_WIDTH,
                        help=f"Bin width for surface density scan, Å (default: {DEFAULT_SCAN_BIN_WIDTH})")
    args = parser.parse_args()

    scan_cutoffs = None
    if args.scan_cutoffs:
        scan_cutoffs = [float(x) for x in args.scan_cutoffs.split(",")]

    site_by_id = None
    if args.dump0:
        from coverage_metrics import classify_initial_sites
        print(f"\nClassifying surface sites from pristine slab: {args.dump0}")
        site_by_id, inventory, _ = classify_initial_sites(
            args.dump0,
            substrate_types=list(SUBSTRATE_TYPES),
            slab_thickness=args.slab_thickness,
            density_fraction=args.density_fraction,
            min_layer_atoms=args.min_layer_atoms,
            scan_bin_width=args.scan_bin_width,
            si_o_cutoff=args.si_o_cutoff,
            o_h_cutoff=args.o_h_cutoff,
        )
        print(f"  Initial site inventory: {inventory}\n")

    run(
        dump_path=args.dump,
        output_path=args.output,
        chem_cutoff=args.chem_cutoff,
        contact_cutoff=args.contact_cutoff,
        al_c_cutoff=args.al_c_cutoff,
        scan_cutoffs=scan_cutoffs,
        stride=args.stride,
        slab_thickness=args.slab_thickness,
        density_fraction=args.density_fraction,
        min_layer_atoms=args.min_layer_atoms,
        scan_bin_width=args.scan_bin_width,
        site_by_id=site_by_id,
    )
