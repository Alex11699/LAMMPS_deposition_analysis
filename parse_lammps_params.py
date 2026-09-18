"""
parse_lammps_params.py
-------------------------
Extract the key deposition/thermostat parameters from a LAMMPS input
script (fix deposit N/type/M/seed/velocity, and the mobile-group Langevin
target temperature), so you can audit what baseline each sweep run
actually held fixed without manually reading every input.lammps by hand.

Usage:
    python parse_lammps_params.py --input input.lammps
    python parse_lammps_params.py --root /path/to/Alex-format-runs-August --output params_audit.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


DEPOSIT_RE = re.compile(
    r"^\s*fix\s+\S+\s+\S+\s+deposit\s+"
    r"(?P<N>\d+)\s+(?P<type>\d+)\s+(?P<M>\d+)\s+(?P<seed>\d+)"
    r"(?P<rest>.*)$"
)
VZ_RE = re.compile(r"vz\s+(?P<vz_lo>-?[\d.]+)\s+(?P<vz_hi>-?[\d.]+)")
LANGEVIN_RE = re.compile(
    r"^\s*fix\s+\S+\s+\S+\s+langevin\s+(?P<t_start>[\d.]+)\s+(?P<t_end>[\d.]+)\s+(?P<damp>[\d.]+)\s+(?P<seed>\d+)"
)
VELOCITY_CREATE_RE = re.compile(r"^\s*velocity\s+\S+\s+create\s+[\d.]+\s+(?P<seed>\d+)")
NVT_RE = re.compile(
    r"^\s*fix\s+\S+\s+\S+\s+nvt\s+temp\s+(?P<t_start>[\d.]+)\s+(?P<t_end>[\d.]+)\s+(?P<damp>[\d.]+)"
)
RUN_RE = re.compile(r"^\s*run\s+(?P<steps>\d+)")


def parse_lammps_input(input_path: str) -> dict:
    """Extract deposit N/type/M/seed/vz, Langevin/NVT target temp, and run lengths."""
    result = {
        "deposit_N": None, "deposit_type": None, "deposit_M": None, "deposit_seed": None,
        "vz_lo": None, "vz_hi": None,
        "langevin_temp": None, "langevin_seed": None, "nvt_temp_end": None,
        "velocity_create_seed": None,
        "run_lengths": [],
    }

    with open(input_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue   # skip commented-out lines — don't pick up disabled alternatives

            m = DEPOSIT_RE.match(line)
            if m:
                result["deposit_N"] = int(m.group("N"))
                result["deposit_type"] = int(m.group("type"))
                result["deposit_M"] = int(m.group("M"))
                result["deposit_seed"] = int(m.group("seed"))
                vzm = VZ_RE.search(m.group("rest"))
                if vzm:
                    result["vz_lo"] = float(vzm.group("vz_lo"))
                    result["vz_hi"] = float(vzm.group("vz_hi"))
                continue

            m = LANGEVIN_RE.match(line)
            if m:
                # Report the END (target) temperature — the value the system is
                # actually held at during deposition, not the ramp start.
                result["langevin_temp"] = float(m.group("t_end"))
                result["langevin_seed"] = int(m.group("seed"))
                continue

            m = NVT_RE.match(line)
            if m:
                result["nvt_temp_end"] = float(m.group("t_end"))
                continue

            m = VELOCITY_CREATE_RE.match(line)
            if m:
                result["velocity_create_seed"] = int(m.group("seed"))
                continue

            m = RUN_RE.match(line)
            if m:
                result["run_lengths"].append(int(m.group("steps")))

    return result


def audit_tree(root: str) -> pd.DataFrame:
    """Find every input.lammps under root and parse it into one summary row per directory."""
    root_path = Path(root)
    rows = []
    for input_path in sorted(root_path.rglob("input.lammps")):
        run_dir = input_path.parent
        rel = run_dir.relative_to(root_path)
        parts = rel.parts
        group = parts[-2] if len(parts) >= 2 else str(rel)
        value = parts[-1] if len(parts) >= 2 else ""

        params = parse_lammps_input(str(input_path))
        rows.append({
            "group": group, "value": value, "run_dir": str(run_dir),
            "deposit_N": params["deposit_N"], "deposit_M": params["deposit_M"],
            "deposit_seed": params["deposit_seed"],
            "vz_lo": params["vz_lo"], "vz_hi": params["vz_hi"],
            "langevin_temp": params["langevin_temp"], "langevin_seed": params["langevin_seed"],
            "nvt_temp_end": params["nvt_temp_end"],
            "velocity_create_seed": params["velocity_create_seed"],
            "run_lengths": ";".join(str(r) for r in params["run_lengths"]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit deposit/thermostat parameters across LAMMPS input scripts")
    group_arg = parser.add_mutually_exclusive_group(required=True)
    group_arg.add_argument("--input", help="Single input.lammps file to parse")
    group_arg.add_argument("--root", help="Root directory to search recursively for input.lammps files")
    parser.add_argument("--output", default=None, help="CSV output path (used with --root)")
    args = parser.parse_args()

    if args.input:
        result = parse_lammps_input(args.input)
        for k, v in result.items():
            print(f"  {k:16s}: {v}")
    else:
        df = audit_tree(args.root)
        print(df.to_string(index=False))
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved -> {args.output}")
