"""
plot_reaction_progress.py
----------------------------
Time-series view of Al reaction progress, built from the per-frame
al_states.csv that deposition_analysis.py already produces — no need to
rerun the (expensive) OVITO trajectory pass.

Shows:
  - Total Al atoms present over time (deposition curve)
  - Fraction in each ligand_state (TMA-like / DMA-like / MMA-like / Al*-bare)
  - Fraction "properly reacted" (DMA-like + chemisorbed/incorporated) —
    the direct measure of how much precursor has completed ligand
    exchange and is ready for the next half-cycle
  - Fraction still unbound/physisorbed (not yet reacted)

Usage:
    python plot_reaction_progress.py --al-states analysis_al_states.csv --output progress.png
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def load_and_aggregate(al_states_path: str) -> pd.DataFrame:
    """
    al_states.csv has columns like 'DMA-like|chemisorbed', 'TMA-like|unbound',
    etc, one row per frame. Aggregate into the summary quantities we want
    to plot, robust to whichever combinations happen to be present.
    """
    df = pd.read_csv(al_states_path, index_col=0)
    df = df.fillna(0)

    # Column names are "ligand_state|binding_state" — split for aggregation
    cols_by_ligand = {}
    cols_by_binding = {}
    reacted_cols = []      # DMA-like + (chemisorbed or incorporated) = "properly reacted"
    unreacted_cols = []    # anything unbound or physisorbed, regardless of ligand state

    for col in df.columns:
        if "|" not in col:
            continue
        ligand, binding = col.split("|", 1)
        cols_by_ligand.setdefault(ligand, []).append(col)
        cols_by_binding.setdefault(binding, []).append(col)
        if ligand == "DMA-like" and binding in ("chemisorbed", "incorporated"):
            reacted_cols.append(col)
        if binding in ("unbound", "physisorbed"):
            unreacted_cols.append(col)

    summary = pd.DataFrame(index=df.index)
    summary["total_Al"] = df.sum(axis=1)
    for ligand, cols in cols_by_ligand.items():
        summary[f"n_{ligand}"] = df[cols].sum(axis=1)
    for binding, cols in cols_by_binding.items():
        summary[f"n_{binding}"] = df[cols].sum(axis=1)
    summary["n_properly_reacted"] = df[reacted_cols].sum(axis=1) if reacted_cols else 0
    summary["n_unreacted"] = df[unreacted_cols].sum(axis=1) if unreacted_cols else 0
    summary["frac_properly_reacted"] = np.where(
        summary["total_Al"] > 0, summary["n_properly_reacted"] / summary["total_Al"], 0.0
    )

    return summary


def plot_progress(summary: pd.DataFrame, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Panel 1: total Al deposited + reacted count over time ---
    axes[0].plot(summary.index, summary["total_Al"], color="#333333", linewidth=1.4, label="Total Al present")
    if "n_properly_reacted" in summary:
        axes[0].plot(summary.index, summary["n_properly_reacted"], color="#2ca02c",
                     linewidth=1.4, label="DMA-like + chemisorbed/incorporated (\"ready\")")
    if "n_unbound" in summary:
        axes[0].plot(summary.index, summary["n_unbound"], color="#d62728",
                     linewidth=1.0, linestyle="--", label="Unbound")
    if "n_physisorbed" in summary:
        axes[0].plot(summary.index, summary["n_physisorbed"], color="#ff7f0e",
                     linewidth=1.0, linestyle="--", label="Physisorbed")
    axes[0].set_ylabel("Al atom count")
    axes[0].set_title("Al population over the deposition trajectory")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: fraction properly reacted (the key ALD progress metric) ---
    axes[1].plot(summary.index, summary["frac_properly_reacted"] * 100, color="#2ca02c", linewidth=1.6)
    axes[1].set_ylabel("% of Al properly reacted\n(DMA-like + chemisorbed/incorporated)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Al reaction-progress time series from al_states.csv")
    parser.add_argument("--al-states", required=True, help="Path to the *_al_states.csv from deposition_analysis.py")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--csv-out", default=None, help="Optional: also save the aggregated summary as CSV")
    args = parser.parse_args()

    summary = load_and_aggregate(args.al_states)
    print(summary.tail(10))

    plot_progress(summary, args.output)

    if args.csv_out:
        summary.to_csv(args.csv_out)
        print(f"Aggregated summary -> {args.csv_out}")
