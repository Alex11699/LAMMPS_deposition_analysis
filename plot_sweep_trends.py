"""
plot_sweep_trends.py
-----------------------
Visualize batch_summary.csv (from run_batch_analysis.py) faceted by
parameter group — coverage % and sticking coefficient vs parameter value,
one panel per group. Numeric-valued groups (temps, freqs, velocities,
precursor-total) get a line+scatter plot; groups with non-numeric or
mixed values (angles, or a group containing a repeat label like
"250-Try1") fall back to a bar chart so nothing gets silently dropped.

Also plots the al_c_trough_cutoff per condition as a diagnostic panel,
so you can see at a glance which conditions the fixed --al-c-cutoff is
least trustworthy for (confidence == 'low', or a cutoff far from the
rest of the sweep).

Usage:
    python plot_sweep_trends.py --summary batch_summary.csv --output sweep_trends.png
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


import re


def _try_numeric(value):
    """
    Parse a value as float if possible. Strips trailing repeat-run suffixes
    like '-Try1', '-Try2' (a leading numeric portion followed by a dash and
    a label) so a repeat of an existing condition still plots at the same
    x-position on a line plot, rather than forcing the whole group into
    the categorical bar-chart fallback just because one label wasn't a
    bare number.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    match = re.match(r"^(-?\d+\.?\d*)(-.+)?$", str(value))
    if match:
        return float(match.group(1))
    return None


def plot_sweep_trends(summary_path: str, output_path: str, control_csv: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_path)
    df = df[df["error"].isna()] if "error" in df.columns else df
    groups = df["group"].unique().tolist()
    n_groups = len(groups)

    control_df = None
    if control_csv:
        control_df = pd.read_csv(control_csv)
        control_df["numeric_value"] = control_df["value"].apply(_try_numeric)

    fig, axes = plt.subplots(2, n_groups, figsize=(4 * n_groups, 8), squeeze=False)

    for col, group in enumerate(groups):
        sub = df[df["group"] == group].copy()
        sub["numeric_value"] = sub["value"].apply(_try_numeric)
        all_numeric = sub["numeric_value"].notna().all()

        ax_cov = axes[0][col]
        ax_cutoff = axes[1][col]

        if all_numeric:
            sub = sub.sort_values("numeric_value")
            x = sub["numeric_value"]

            ax_cov.plot(x, sub["final_coverage_pct"], "o-", color="#2ca02c", label="Coverage %")
            ax_cov2 = ax_cov.twinx()
            ax_cov2.plot(x, sub["final_sticking_coefficient"], "s--", color="#1f77b4", label="Sticking coeff.")
            ax_cov2.set_ylabel("Sticking coefficient", color="#1f77b4")
            ax_cov.set_ylabel("Coverage %", color="#2ca02c")

            colors = {"exact_zero": "#2ca02c", "near_zero": "#ff7f0e", "low": "#d62728"}
            bar_colors = [colors.get(c, "#888888") for c in sub["al_c_trough_confidence"]]
            ax_cutoff.bar(x, sub["al_c_trough_cutoff"], width=(x.max()-x.min())*0.08 if len(x) > 1 else 0.5,
                          color=bar_colors)
            ax_cutoff.set_xlabel(group)

            # Overlay the control point, if one applies to this group — plotted
            # as a distinct star marker, NOT connected into the sweep line,
            # since it's a different (all-parameters-at-baseline) condition,
            # not one more point along this single 1D sweep.
            if control_df is not None and group in control_df["group"].values:
                crow = control_df[control_df["group"] == group].iloc[0]
                cx = crow["numeric_value"]
                ax_cov.plot(cx, crow["final_coverage_pct"], marker="*", markersize=18,
                            color="#2ca02c", markeredgecolor="black", markeredgewidth=1, linestyle="none", zorder=6)
                ax_cov2.plot(cx, crow["final_sticking_coefficient"], marker="*", markersize=18,
                             color="#1f77b4", markeredgecolor="black", markeredgewidth=1, linestyle="none", zorder=6)
                if "al_c_trough_cutoff" in crow and pd.notna(crow["al_c_trough_cutoff"]):
                    ax_cutoff.plot(cx, crow["al_c_trough_cutoff"], marker="*", markersize=18,
                                   color="gold", markeredgecolor="black", markeredgewidth=1, zorder=6)
        else:
            # Categorical fallback: bar chart, x-axis = value labels directly
            x_labels = sub["value"].astype(str).tolist()
            x_pos = np.arange(len(x_labels))

            ax_cov.bar(x_pos - 0.2, sub["final_coverage_pct"], width=0.4, color="#2ca02c", label="Coverage %")
            ax_cov2 = ax_cov.twinx()
            ax_cov2.bar(x_pos + 0.2, sub["final_sticking_coefficient"], width=0.4, color="#1f77b4", label="Sticking coeff.")
            ax_cov.set_xticks(x_pos)
            ax_cov.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            ax_cov.set_ylabel("Coverage %", color="#2ca02c")
            ax_cov2.set_ylabel("Sticking coefficient", color="#1f77b4")

            colors = {"exact_zero": "#2ca02c", "near_zero": "#ff7f0e", "low": "#d62728"}
            bar_colors = [colors.get(c, "#888888") for c in sub["al_c_trough_confidence"]]
            ax_cutoff.bar(x_pos, sub["al_c_trough_cutoff"], color=bar_colors)
            ax_cutoff.set_xticks(x_pos)
            ax_cutoff.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            ax_cutoff.set_xlabel(group)

        ax_cov.set_title(group)
        ax_cutoff.set_ylabel("Al-C trough cutoff (Å)")
        ax_cutoff.axhspan(2.3, 2.5, color="grey", alpha=0.15, label="stable range seen elsewhere")
        ax_cutoff.grid(True, alpha=0.3)

    # Shared legend for confidence colors, once
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elems = [
        Patch(facecolor="#2ca02c", label="exact_zero"),
        Patch(facecolor="#ff7f0e", label="near_zero"),
        Patch(facecolor="#d62728", label="low confidence"),
    ]
    if control_df is not None:
        legend_elems.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="grey",
                                    markeredgecolor="black", markersize=14, label="control run"))
    fig.legend(handles=legend_elems, loc="lower center", ncol=len(legend_elems), bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle("Coverage / sticking / Al-C cutoff stability across parameter sweep", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep trends from batch_summary.csv")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-csv", default=None,
                         help="Optional CSV of a shared control/baseline run to overlay on each relevant "
                              "panel as a distinct marker. Columns: group,value,final_coverage_pct,"
                              "final_sticking_coefficient,al_c_trough_cutoff — one row per group the "
                              "control condition should appear on (e.g. one row for 'freqs', one for "
                              "'temps', etc., all sharing the same underlying run's metric values).")
    args = parser.parse_args()
    plot_sweep_trends(args.summary, args.output, args.control_csv)
