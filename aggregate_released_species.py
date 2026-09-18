"""
aggregate_released_species.py
--------------------------------
Turns the per-frame species table (analysis.raw.csv, with anchor_key from
the updated deposition_analysis.py) into true "how many distinct
molecules were produced" counts, and aggregates that across a parameter
sweep for cross-condition comparison.

Why this exists: the per-frame species table has no memory between
frames — the same physical CH4 molecule sitting around for 200 frames
gets counted as "1 CH4" in each of those 200 rows. Summing across frames
would report "200 CH4 events" for one molecule. This module deduplicates
by (species, anchor_key), keeping only the first frame each distinct
fragment identity appears as a given species, so the counts mean "N
distinct molecules were produced", not "N frame-appearances".

Usage (single run):
    python aggregate_released_species.py --raw-csv analysis.raw.csv --output species_produced.csv

Usage (bulk, across a sweep — reads each run_dir's analysis.raw.csv from
a batch_summary.csv or params_audit.csv style table):
    python aggregate_released_species.py --batch-table batch_summary.csv \\
        --output species_by_condition.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from deposition_analysis import KNOWN_SPECIES


RELEASED_STATES = ("unbound", "physisorbed")

CANONICAL_SPECIES = set(KNOWN_SPECIES.values())


def bucket_species(species: str, anchor_key: str) -> str:
    """
    Collapse the long tail of exotic composite formulas (e.g. Al3C7H14,
    Al2C6H16 — transient multi-fragment contacts, not distinct chemical
    species) into a small number of catch-all buckets, so bulk comparison
    across a sweep has ~15 columns instead of 500+ mostly-empty ones.
    Canonical, chemically-named species (TMA, DMA*, MMA*, Al*, CH3, CH4,
    H2, H*, TMA-dimer, DMA-O*, C2H6 — from deposition_analysis.KNOWN_SPECIES)
    pass through unchanged.
    """
    if species in CANONICAL_SPECIES:
        return species
    if anchor_key.startswith("Al:"):
        return "other_Al_fragment"
    if anchor_key.startswith("C:"):
        return "other_CH_fragment"
    return "other_H_fragment"


def deduplicate_species(raw_df: pd.DataFrame, states: tuple = RELEASED_STATES,
                         use_buckets: bool = True) -> pd.DataFrame:
    """
    Given the raw per-frame fragment table (frame, species, state,
    anchor_key, ...), return one row per DISTINCT (species, anchor_key)
    that was ever seen in `states`, at its first-appearance frame — this
    is the deduplicated "N molecules produced" table. Non-canonical
    species are bucketed (see bucket_species) before deduplication, so
    an anchor that drifts between several exotic composite labels still
    only counts once under its catch-all bucket.
    """
    if raw_df.empty or "anchor_key" not in raw_df.columns:
        return pd.DataFrame(columns=["species", "anchor_key", "first_frame", "last_frame", "n_frames_present"])

    released = raw_df[raw_df["state"].isin(states)].copy()
    if released.empty:
        return pd.DataFrame(columns=["species", "anchor_key", "first_frame", "last_frame", "n_frames_present"])

    if use_buckets:
        released["species"] = [bucket_species(s, a) for s, a in zip(released["species"], released["anchor_key"])]

    grouped = released.groupby(["species", "anchor_key"])["frame"].agg(
        first_frame="min", last_frame="max", n_frames_present="count"
    ).reset_index()

    return grouped.sort_values("first_frame")


def summarize_run(raw_csv_path: str) -> pd.DataFrame:
    """Deduplicated per-species production counts for one run."""
    raw_df = pd.read_csv(raw_csv_path)
    dedup = deduplicate_species(raw_df)
    if dedup.empty:
        return pd.DataFrame(columns=["species", "n_produced"])
    summary = dedup.groupby("species").size().reset_index(name="n_produced")
    return summary.sort_values("n_produced", ascending=False)


def aggregate_bulk(batch_table_path: str, raw_csv_name: str = "analysis.raw.csv") -> pd.DataFrame:
    """
    For every run_dir listed in a batch summary table (from
    run_batch_analysis.py or params_audit.csv), find its raw fragment
    CSV, deduplicate, and pivot into one wide row per condition:
    group, value, run_dir, <species columns...>.
    """
    batch_df = pd.read_csv(batch_table_path)
    rows = []

    for _, row in batch_df.iterrows():
        run_dir = row.get("run_dir")
        if not run_dir or pd.isna(run_dir):
            continue
        raw_csv_path = Path(run_dir) / raw_csv_name
        if not raw_csv_path.exists():
            print(f"  SKIPPING {run_dir} — no {raw_csv_name} found")
            continue

        summary = summarize_run(str(raw_csv_path))
        out_row = {"group": row.get("group"), "value": row.get("value"), "run_dir": run_dir}
        for _, s in summary.iterrows():
            out_row[f"n_{s['species']}"] = int(s["n_produced"])
        rows.append(out_row)

    result = pd.DataFrame(rows).fillna(0)
    # Cast species count columns to int (fillna introduces floats)
    for col in result.columns:
        if col.startswith("n_"):
            result[col] = result[col].astype(int)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate and aggregate released-species production counts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--raw-csv", help="Single run's analysis.raw.csv")
    group.add_argument("--batch-table", help="batch_summary.csv or similar, with a run_dir column, "
                                              "to aggregate across many runs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.raw_csv:
        result = summarize_run(args.raw_csv)
    else:
        result = aggregate_bulk(args.batch_table)

    print(result.to_string(index=False))
    result.to_csv(args.output, index=False)
    print(f"\nSaved -> {args.output}")
