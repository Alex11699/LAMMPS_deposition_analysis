"""
run_batch_analysis.py
------------------------
Walks a directory tree of parameter-sweep LAMMPS runs (e.g.
Alex-format-runs-August/<group>/<value>/dump1.lammpstrj), runs the full
deposition analysis pipeline on each, and aggregates results into one
master summary CSV for cross-condition comparison.

For each run directory found (identified by containing dump1.lammpstrj,
with dump0.lammpstrj alongside for OH-site counting):
  1. Run deposition_analysis.run() with the agreed cutoffs
     (--chem-cutoff, --al-c-cutoff) — writes the usual per-run output
     files (analysis.csv, analysis_al_atoms.raw.csv, etc.) into that
     same directory, exactly as running deposition_analysis.py directly
     would.
  2. Count initial OH sites from dump0.lammpstrj and compute coverage %
     / sticking coefficient time series (coverage_metrics.py).
  3. Sample a handful of late frames and auto-detect the Al-C bonding
     cutoff trough (al_c_distance_histogram.find_bonding_cutoff) — this
     is the "does the cutoff choice still hold up under different
     conditions" check, done automatically instead of re-reading every
     histogram by hand.
  4. Append one summary row (final-frame numbers + cutoff-stability
     check) to the master CSV.

Usage:
    python run_batch_analysis.py --root /path/to/Alex-format-runs-August \\
        --output batch_summary.csv --chem-cutoff 2.2 --al-c-cutoff 2.4 --stride 5

Parameter labels are taken from the two path components directly above
dump1.lammpstrj (e.g. "temps/400" -> group="temps", value="400"). Adjust
--group-depth if your directory structure nests differently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from deposition_analysis import run as run_deposition_analysis
from coverage_metrics import (
    count_oh_sites, compute_coverage_sticking, tail_average,
    classify_initial_sites, compute_site_type_coverage,
)
from al_c_distance_histogram import collect_al_c_distances, find_bonding_cutoff


def find_run_directories(root: str) -> list[Path]:
    """Find every directory under root containing both dump0 and dump1 dumps."""
    root_path = Path(root)
    run_dirs = []
    for dump1_path in root_path.rglob("dump1.lammpstrj"):
        run_dir = dump1_path.parent
        if (run_dir / "dump0.lammpstrj").exists():
            run_dirs.append(run_dir)
        else:
            print(f"  SKIPPING {run_dir} — dump1.lammpstrj found but no matching dump0.lammpstrj")
    return sorted(run_dirs)


def extract_param_label(run_dir: Path, root: Path, group_depth: int = 2) -> tuple:
    """
    Extract (group, value) from the run directory's path relative to root.
    E.g. root/temps/400 -> ("temps", "400").

    Handles an extra nesting level for repeat/named-variant runs, e.g.
    root/freqs/250/Try1 -> naive parts[-2:] gives ("250", "Try1"), which
    is wrong — "250" is a value, not a group name, and this run belongs
    to the "freqs" sweep. Detected by checking whether parts[-2] parses
    as a plain number (a group name never should); if so, walk up one
    more level for the true group and combine the extra segment into the
    value as "<value>-<variant>" (e.g. "250-Try1"), matching the
    convention used elsewhere for repeat runs.

    Falls back to the full relative path string if the tree is shallower
    than group_depth.
    """
    rel = run_dir.relative_to(root)
    parts = rel.parts

    if len(parts) < group_depth:
        return str(rel), ""

    group, value = parts[-2], parts[-1]
    if _looks_numeric(group) and len(parts) >= group_depth + 1:
        group = parts[-3]
        value = f"{parts[-2]}-{parts[-1]}"

    return group, value


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def run_batch(
    root: str,
    output_path: str,
    chem_cutoff: float,
    al_c_cutoff: float,
    contact_cutoff: float = 4.0,
    stride: int = 5,
    late_frame_sample_count: int = 5,
    hydroxyl_type: int = 1,
    avg_window: int = 20,
    classify_sites: bool = False,
    si_o_cutoff: float = 2.0,
    o_h_cutoff: float = 1.2,
):
    root_path = Path(root)
    run_dirs = find_run_directories(root)
    print(f"Found {len(run_dirs)} run directories under {root}\n")

    summary_rows = []

    for i, run_dir in enumerate(run_dirs):
        group, value = extract_param_label(run_dir, root_path)
        print(f"[{i+1}/{len(run_dirs)}] {group}/{value}  ({run_dir})")

        dump1_path = str(run_dir / "dump1.lammpstrj")
        dump0_path = str(run_dir / "dump0.lammpstrj")
        output_csv = str(run_dir / "analysis.csv")

        # Site classification (opt-in via --classify-sites) — needs to happen
        # BEFORE the main analysis run so site_by_id can be threaded through
        # to al_bonding_analysis (bound_site_type column).
        site_by_id, site_inventory = None, None
        if classify_sites:
            try:
                site_by_id, site_inventory, _ = classify_initial_sites(
                    dump0_path, si_o_cutoff=si_o_cutoff, o_h_cutoff=o_h_cutoff,
                )
            except Exception as e:
                print(f"  WARNING: site classification failed: {e}")

        try:
            df, summary, species_over_time, al_df, al_state_counts = run_deposition_analysis(
                dump_path=dump1_path,
                output_path=output_csv,
                chem_cutoff=chem_cutoff,
                contact_cutoff=contact_cutoff,
                al_c_cutoff=al_c_cutoff,
                stride=stride,
                site_by_id=site_by_id,
            )
        except Exception as e:
            print(f"  ERROR running deposition_analysis: {e}")
            summary_rows.append({"group": group, "value": value, "run_dir": str(run_dir), "error": str(e)})
            continue

        # OH site count + coverage/sticking, averaged over the last avg_window
        # analysed frames rather than the single final frame — a single
        # endpoint frame is sensitive to whatever noise it happened to land
        # on (see the ~10pp control-run spread noted in project learnings).
        try:
            n_oh, sz0 = count_oh_sites(dump0_path, hydroxyl_type=hydroxyl_type)
            cov_df = compute_coverage_sticking(al_df, n_oh)
            cov_df.to_csv(run_dir / "coverage_sticking.csv", index=False)
            final_cov = tail_average(
                cov_df, ["n_bonded", "coverage_pct", "sticking_coefficient"], n_frames=avg_window
            ) if not cov_df.empty else None
        except Exception as e:
            print(f"  WARNING: coverage/sticking calc failed: {e}")
            n_oh, final_cov = None, None

        # Per-site-type coverage fractions (opt-in) — kept as separate
        # frac_<site_type> columns rather than one combined number; see
        # coverage_metrics.compute_site_type_coverage docstring for why.
        site_type_cov = {}
        if classify_sites and site_inventory:
            try:
                site_cov_df = compute_site_type_coverage(al_df, site_inventory)
                site_cov_df.to_csv(run_dir / "site_type_coverage.csv", index=False)
                frac_cols = [c for c in site_cov_df.columns if c.startswith("frac_")]
                if not site_cov_df.empty and frac_cols:
                    site_type_cov = tail_average(site_cov_df, frac_cols, n_frames=avg_window)
            except Exception as e:
                print(f"  WARNING: per-site-type coverage calc failed: {e}")

        # Al-C cutoff stability check on late frames
        try:
            from ovito.io import import_file
            pipeline = import_file(dump1_path, multiple_frames=True)
            n_frames = pipeline.source.num_frames
            late_frames = list(range(max(0, n_frames - late_frame_sample_count), n_frames))
            al_c_dists = collect_al_c_distances(dump1_path, late_frames, al_type=4, carbon_type=5)
            trough = find_bonding_cutoff(al_c_dists)
        except Exception as e:
            print(f"  WARNING: Al-C cutoff check failed: {e}")
            trough = None

        row = {
            "group": group, "value": value, "run_dir": str(run_dir),
            "n_oh_sites_initial": n_oh,
            "final_n_Al_total": int(al_df[al_df["frame"] == al_df["frame"].max()].shape[0]) if not al_df.empty else 0,
            # "final_*" fields are now means over the last avg_window analysed
            # frames (see tail_average), not a single last-frame value.
            "avg_window_n_frames": final_cov["n_frames_averaged"] if final_cov is not None else None,
            "avg_window_frame_range": final_cov["frame_range"] if final_cov is not None else None,
            "final_n_bonded": final_cov["n_bonded"] if final_cov is not None else None,
            "final_coverage_pct": final_cov["coverage_pct"] if final_cov is not None else None,
            "final_coverage_pct_std": final_cov["coverage_pct_std"] if final_cov is not None else None,
            "final_sticking_coefficient": final_cov["sticking_coefficient"] if final_cov is not None else None,
            "final_sticking_coefficient_std": final_cov["sticking_coefficient_std"] if final_cov is not None else None,
            "al_c_trough_cutoff": trough["cutoff"] if trough else None,
            "al_c_trough_confidence": trough["confidence"] if trough else None,
            "al_c_trough_min_count": trough["min_count"] if trough else None,
            "al_c_trough_peak_count": trough["peak_count"] if trough else None,
        }
        if site_inventory:
            for site_type, n_initial in site_inventory.items():
                row[f"n_{site_type}_initial"] = n_initial
                row[f"final_frac_{site_type}"] = site_type_cov.get(f"frac_{site_type}")
        summary_rows.append(row)
        cov_str = f"{row['final_coverage_pct']:.2f}%" if row['final_coverage_pct'] is not None else "n/a"
        stick_str = f"{row['final_sticking_coefficient']:.4f}" if row['final_sticking_coefficient'] is not None else "n/a"
        print(f"  -> coverage={cov_str} (avg over {row['avg_window_n_frames']} frames)  sticking={stick_str}  "
              f"al_c_trough={row['al_c_trough_cutoff']} (confidence={row['al_c_trough_confidence']})\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    print(f"\nBatch summary -> {output_path}")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-run deposition_analysis.py across a parameter sweep directory tree")
    parser.add_argument("--root", required=True, help="Root directory to search for dump1.lammpstrj files")
    parser.add_argument("--output", required=True, help="Master summary CSV output path")
    parser.add_argument("--chem-cutoff", type=float, default=2.2)
    parser.add_argument("--al-c-cutoff", type=float, default=2.4)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--stride", type=int, default=5, help="Frame stride for the main analysis (default: 5, "
                         "coarser than single-run default of 1 since this runs many trajectories)")
    parser.add_argument("--late-frame-sample-count", type=int, default=5,
                         help="Number of trailing frames to sample for the Al-C cutoff stability check (default: 5)")
    parser.add_argument("--hydroxyl-type", type=int, default=1)
    parser.add_argument("--avg-window", type=int, default=20,
                         help="Number of trailing analysed frames to average coverage/sticking/"
                              "site-type-coverage over, instead of taking the single final frame "
                              "(default: 20 analysed frames, i.e. after --stride)")
    parser.add_argument("--classify-sites", action="store_true",
                         help="Classify surface O/Si sites (hydroxyl_O/bridging_O/dangling_O/"
                              "*_Si) from each run's dump0 and compute per-site-type coverage "
                              "fractions alongside the existing hydroxyl-only coverage_pct. "
                              "Off by default — opt in once the cutoffs below are validated.")
    parser.add_argument("--si-o-cutoff", type=float, default=2.0,
                         help="Si-O bonding cutoff for site classification, Å (default: 2.0)")
    parser.add_argument("--o-h-cutoff", type=float, default=1.2,
                         help="O-H bonding cutoff for site classification, Å (default: 1.2)")
    args = parser.parse_args()

    run_batch(
        root=args.root, output_path=args.output,
        chem_cutoff=args.chem_cutoff, al_c_cutoff=args.al_c_cutoff,
        contact_cutoff=args.contact_cutoff, stride=args.stride,
        late_frame_sample_count=args.late_frame_sample_count,
        avg_window=args.avg_window,
        classify_sites=args.classify_sites,
        si_o_cutoff=args.si_o_cutoff, o_h_cutoff=args.o_h_cutoff,
        hydroxyl_type=args.hydroxyl_type,
    )
