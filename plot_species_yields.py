"""
plot_species_yields.py
-------------------------
Visualize species_by_condition.csv (from aggregate_released_species.py)
as per-TMA-delivered yield ratios, faceted by parameter group — same
numeric-vs-categorical handling as plot_sweep_trends.py, but showing
several species' yields together on each panel (log y-axis, since yields
span ~2-3 orders of magnitude across the sweep) rather than one quantity
on twin axes.

Usage:
    python plot_species_yields.py --species-csv species_by_condition.csv --output species_yields.png
    python plot_species_yields.py --species-csv species_by_condition.csv --output species_yields.png \\
        --species CH4 C2H6 H2 other_Al_fragment TMA-dimer
"""

from __future__ import annotations

import argparse
import re
import numpy as np
import pandas as pd


DEFAULT_SPECIES = ["CH4", "C2H6", "H2", "other_Al_fragment", "TMA-dimer"]
SPECIES_COLORS = {
    "CH4": "#2ca02c", "C2H6": "#d62728", "H2": "#1f77b4",
    "other_Al_fragment": "#9467bd", "TMA-dimer": "#ff7f0e",
    "DMA*": "#17becf", "MMA*": "#8c564b", "Al*": "#7f7f7f",
    "H*": "#e377c2", "CH3": "#bcbd22", "other_CH_fragment": "#aec7e8",
    "other_H_fragment": "#c5b0d5",
}


def _try_numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    match = re.match(r"^(-?\d+\.?\d*)(-.+)?$", str(value))
    if match:
        return float(match.group(1))
    return None


def plot_species_yields(species_csv: str, output_path: str, species: list = None, log_scale: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    species = species or DEFAULT_SPECIES
    df = pd.read_csv(species_csv)

    if "n_TMA" not in df.columns:
        raise ValueError("species_by_condition.csv must include an 'n_TMA' column to normalize yields against.")

    # Compute yields fresh (don't assume the CSV already has yield_ columns)
    for s in species:
        col = f"n_{s}"
        if col in df.columns:
            df[f"yield_{s}"] = df[col] / df["n_TMA"]

    groups = df["group"].unique().tolist()
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(4.2 * n_groups, 5), squeeze=False)
    axes = axes[0]

    for col, group in enumerate(groups):
        sub = df[df["group"] == group].copy()
        sub["numeric_value"] = sub["value"].apply(_try_numeric)
        all_numeric = sub["numeric_value"].notna().all()
        ax = axes[col]

        if len(sub) == 1:
            # A single-point group (e.g. a standalone baseline run) can't
            # show a trend — render as a simple labeled bar per species
            # instead of a line plot with a meaningless numeric x-axis.
            x_pos = np.arange(len(species))
            heights = [sub[f"yield_{s}"].iloc[0] if f"yield_{s}" in sub.columns else 0 for s in species]
            colors = [SPECIES_COLORS.get(s, None) for s in species]
            ax.bar(x_pos, heights, color=colors)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(species, rotation=30, ha="right", fontsize=8)
            ax.set_xlabel(group)
        elif all_numeric:
            sub = sub.sort_values("numeric_value")
            x = sub["numeric_value"]
            for s in species:
                ycol = f"yield_{s}"
                if ycol in sub.columns:
                    ax.plot(x, sub[ycol], "o-", color=SPECIES_COLORS.get(s, None), label=s, markersize=5)
            ax.set_xlabel(group)
        else:
            x_labels = sub["value"].astype(str).tolist()
            x_pos = np.arange(len(x_labels))
            width = 0.8 / max(len(species), 1)
            for i, s in enumerate(species):
                ycol = f"yield_{s}"
                if ycol in sub.columns:
                    ax.bar(x_pos + i * width, sub[ycol], width=width, color=SPECIES_COLORS.get(s, None), label=s)
            ax.set_xticks(x_pos + width * (len(species) - 1) / 2)
            ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            ax.set_xlabel(group)

        if log_scale:
            ax.set_yscale("log")
        ax.set_title(group)
        ax.grid(True, alpha=0.3, which="both")
        if col == 0:
            ax.set_ylabel("Yield (molecules produced per TMA delivered)" + (" [log scale]" if log_scale else ""))

    # One shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        # angles-only panels might be first; grab from any panel that has them
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="lower center", ncol=len(species), bbox_to_anchor=(0.5, -0.05), fontsize=9)

    fig.suptitle("Byproduct species yield per TMA delivered, across parameter sweep", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot species yield trends from species_by_condition.csv")
    parser.add_argument("--species-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--species", nargs="+", default=None,
                         help=f"Species to plot (default: {DEFAULT_SPECIES})")
    parser.add_argument("--linear", action="store_true", help="Use linear y-axis instead of the log-scale default")
    args = parser.parse_args()

    plot_species_yields(args.species_csv, args.output, args.species, log_scale=not args.linear)
