"""
al_c_distance_histogram.py
-----------------------------
Builds the actual Al-C pairwise distance distribution from the trajectory,
so --al-c-cutoff can be chosen from the real bond-length peak rather than
guessed. Same principle as detect_surface_z: read the real distribution
instead of assuming a threshold.

Early frames are the best reference — TMA molecules that have just
landed are mostly intact, so their Al-C distances show a clean, narrow
peak at the true equilibrium bond length before reaction chemistry
starts stretching/breaking bonds. Late frames are also worth checking
separately (a second run) to see whether reacted Al-C bonds shift.

Usage:
    python al_c_distance_histogram.py --dump dump1.lammpstrj --frames 0,10,20,30,40,50
    python al_c_distance_histogram.py --dump dump1.lammpstrj --frames 0,10,20 --plot histogram.png
    python al_c_distance_histogram.py --dump dump1.lammpstrj --frames 990,995,999 --plot late.png
"""

from __future__ import annotations

import argparse
import numpy as np
from ovito.io import import_file
from ovito.data import CutoffNeighborFinder


def collect_al_c_distances(dump_path: str, frames: list[int], al_type: int,
                            carbon_type: int, max_range: float = 6.0) -> np.ndarray:
    """
    For each given frame, find every Al-C pair distance up to max_range
    (generous — well beyond any plausible bonding cutoff) and pool them
    across all requested frames into one array.
    """
    pipeline = import_file(dump_path, multiple_frames=True)
    n_frames = pipeline.source.num_frames
    all_dists = []

    for frame in frames:
        if frame >= n_frames:
            print(f"  (skipping frame {frame}, only {n_frames} frames in trajectory)")
            continue
        data = pipeline.compute(frame)
        types = np.array(data.particles["Particle Type"][:])
        al_indices = np.where(types == al_type)[0]

        finder = CutoffNeighborFinder(max_range, data)
        for idx in al_indices:
            for neigh in finder.find(idx):
                if types[neigh.index] == carbon_type:
                    all_dists.append(neigh.distance)   # read immediately — iterator is transient

    return np.array(all_dists)


def find_bonding_cutoff(dists: np.ndarray, bin_width: float = 0.05,
                         peak_search_range: tuple = (1.5, 2.5),
                         max_search: float = 4.0,
                         near_zero_abs_threshold: int = 2) -> dict | None:
    """
    Automatically locate the first genuine minimum after the Al-C bonding
    peak, for checking cutoff stability across many runs without manually
    reading each histogram.

    Finds the peak bin within peak_search_range, then scans forward for
    the first bin at or below near_zero_abs_threshold (default: 2) counts.
    An exact zero is preferred and reported separately when found, but a
    small ABSOLUTE count threshold (not a fraction of peak height) is used
    as the acceptance criterion, because requiring a literal zero is
    fragile: with a continuous underlying distribution, a single stray
    pair landing in the true minimum bin by ordinary sampling variance —
    especially likely with more total pairs pooled — flips an "exact
    zero" result to nonzero without the real minimum having moved at all.
    A relative/fractional threshold (e.g. "5% of peak") was tried first
    and rejected for the opposite failure mode: on a tall, sharply-peaked
    distribution it triggers on the decaying tail well before the real
    minimum, since the noise floor doesn't scale linearly with peak height.

    If no bin at or below the threshold is found within max_search (a
    genuinely ambiguous or bimodal distribution), falls back to the
    single lowest-count bin in the window and flags confidence as "low"
    so it's not mistaken for a clean answer.

    Returns a dict with:
      cutoff        : suggested cutoff (Å)
      min_count     : the actual count in that bin
      peak_count    : count at the bonding peak, for context
      confidence    : "exact_zero" | "near_zero" | "low"
    or None if no clear peak is found at all (e.g. too few pairs).
    """
    if len(dists) < 10:
        return None

    bins = np.arange(0, max_search + bin_width, bin_width)
    counts, edges = np.histogram(dists, bins=bins)

    lo_idx = int(peak_search_range[0] / bin_width)
    hi_idx = int(peak_search_range[1] / bin_width)
    if hi_idx <= lo_idx or hi_idx > len(counts):
        return None

    peak_idx = lo_idx + int(np.argmax(counts[lo_idx:hi_idx]))
    peak_count = int(counts[peak_idx])
    if peak_count == 0:
        return None

    for i in range(peak_idx, len(counts)):
        if counts[i] == 0:
            return {"cutoff": float(edges[i]), "min_count": 0,
                    "peak_count": peak_count, "confidence": "exact_zero"}
        if counts[i] <= near_zero_abs_threshold:
            return {"cutoff": float(edges[i]), "min_count": int(counts[i]),
                    "peak_count": peak_count, "confidence": "near_zero"}

    # No exact/near-zero bin found — fall back to the lowest-count bin after
    # the peak, flagged as low confidence rather than a clean answer.
    search_end = min(len(counts), int(max_search / bin_width))
    min_idx = peak_idx + int(np.argmin(counts[peak_idx:search_end]))
    return {"cutoff": float(edges[min_idx]), "min_count": int(counts[min_idx]),
            "peak_count": peak_count, "confidence": "low"}


def print_histogram(dists: np.ndarray, bin_width: float = 0.05, max_range: float = 6.0):
    bins = np.arange(0, max_range + bin_width, bin_width)
    counts, edges = np.histogram(dists, bins=bins)
    max_count = counts.max() if counts.size and counts.max() > 0 else 1

    print(f"\nAl-C pairwise distance histogram ({len(dists)} pairs, {bin_width} Å bins):")
    # Only print non-empty region plus a little padding, so it's not 120 lines of zeros
    nonzero = np.where(counts > 0)[0]
    if len(nonzero) == 0:
        print("  No Al-C pairs found within max_range.")
        return
    lo, hi = max(0, nonzero[0] - 3), min(len(counts), nonzero[-1] + 4)
    for i in range(lo, hi):
        bar = "#" * round(50 * counts[i] / max_count)
        print(f"  [{edges[i]:5.2f}, {edges[i+1]:5.2f})  {counts[i]:5d}  {bar}")

    print(f"\nLook for: a narrow peak (the real Al-C bond length) followed by a clear DROP "
          f"to near-zero, then possibly a second, broader peak further out (non-bonded "
          f"contacts). Pick --al-c-cutoff at the bottom of that first drop/minimum — "
          f"not in the middle of the peak, and not so far out that you're back in the "
          f"second peak.")


def plot_histogram(dists: np.ndarray, output_path: str, bin_width: float = 0.05, max_range: float = 6.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.arange(0, max_range + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dists, bins=bins, color="#2ca02c", edgecolor="none")
    ax.set_xlabel("Al-C distance (Å)")
    ax.set_ylabel("Pair count")
    ax.set_title(f"Al-C distance distribution ({len(dists)} pairs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the real Al-C distance distribution for cutoff selection")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--frames", required=True, help="Comma-separated frame indices, e.g. '0,10,20,30'")
    parser.add_argument("--al-type", type=int, default=4)
    parser.add_argument("--carbon-type", type=int, default=5)
    parser.add_argument("--max-range", type=float, default=6.0, help="Upper distance bound to scan, Å (default: 6.0)")
    parser.add_argument("--bin-width", type=float, default=0.05, help="Histogram bin width, Å (default: 0.05)")
    parser.add_argument("--plot", type=str, default=None, help="Optional PNG output path")
    args = parser.parse_args()

    frames = [int(f) for f in args.frames.split(",")]
    print(f"Collecting Al-C distances from frames: {frames}")

    dists = collect_al_c_distances(args.dump, frames, args.al_type, args.carbon_type, args.max_range)
    print_histogram(dists, args.bin_width, args.max_range)

    if args.plot:
        plot_histogram(dists, args.plot, args.bin_width, args.max_range)
