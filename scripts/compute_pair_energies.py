#!/usr/bin/env python3
"""Create energies_{pair}_{box}nm_sliced.pkl files from full run trajectories.

This script builds per-pair energy pickle files directly from the original
CALVADOS run folders, not from center-of-mass combined trajectories.

Expected run layout
-------------------
run/
  test_<pair_id>/
    298k_20.0nm/
      job_0/
      job_1/
      ...

Inside each job folder, the script looks for one simulation PDB and one DCD.
By default it searches for files matching:
- *_job_*.pdb
- *.dcd

Outputs
-------
- one pickle per pair:
    <output_dir>/energies_<pair_id>_<box_label>_sliced.pkl

The pickle contains a dictionary with keys:
- name
- box_nm
- temp
- ionic
- n_jobs
- avg_ah_per_pair
- avg_yu_per_pair
- avg_total_per_pair
- pairs_array
- res1
- res2
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

from per_pair_energies import calc_per_pair_average_energy


REQUIRED_PAIR_COLUMNS = {"pair_id", "idr_a", "idr_b"}


class EnergyBuildError(Exception):
    pass


@dataclass
class JobInput:
    job_id: str
    pdb_file: Path
    dcd_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-pair energy pickle files from full CALVADOS run folders."
    )
    parser.add_argument("--pairs-csv", required=True, type=Path, help="Canonical pair table CSV.")
    parser.add_argument("--runs-dir", required=True, type=Path, help="Root directory containing test_<pair_id> run folders.")
    parser.add_argument("--residues-file", required=True, type=Path, help="Residues CSV for CALVADOS parameters.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for energy pickle outputs.")
    parser.add_argument("--pair-dir-prefix", default="test_", help="Prefix used for pair folders under runs-dir. Default: test_.")
    parser.add_argument("--temp-dir-name", default="298k_20.0nm", help="Temperature/box subdirectory name. Default: 298k_20.0nm.")
    parser.add_argument("--job-dir-glob", default="job_*", help="Glob used to discover job folders. Default: job_*.")
    parser.add_argument("--pdb-pattern", default="*_job_*.pdb", help="Glob used to find the simulation PDB within each job folder. Default: *_job_*.pdb.")
    parser.add_argument("--dcd-pattern", default="*.dcd", help="Glob used to find the trajectory DCD within each job folder. Default: *.dcd.")
    parser.add_argument("--box-label", default="20nm", help="Filename box label used in output names. Default: 20nm.")
    parser.add_argument("--box-nm", type=float, default=20.0, help="Numeric box size in nm. Default: 20.0.")
    parser.add_argument("--temp", type=float, default=298.0, help="Temperature in K. Default: 298.0.")
    parser.add_argument("--ionic", type=float, default=0.15, help="Ionic strength in M. Default: 0.15.")
    parser.add_argument("--eps-lj", type=float, default=0.8368, help="Ashbaugh-Hatch epsilon in kJ/mol. Default: 0.8368.")
    parser.add_argument("--rc-lj", type=float, default=2.0, help="AH cutoff in nm. Default: 2.0.")
    parser.add_argument("--rc-yu", type=float, default=4.0, help="Yukawa cutoff in nm. Default: 4.0.")
    parser.add_argument("--min-jobs", type=int, default=1, help="Minimum number of valid job folders required per pair. Default: 1.")
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise EnergyBuildError(f"{label} not found: {path}")
    if not path.is_file():
        raise EnergyBuildError(f"{label} is not a file: {path}")


def read_pairs(path: Path) -> list[dict[str, str]]:
    ensure_file(path, "Pairs CSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EnergyBuildError(f"Pairs CSV has no header: {path}")
        missing = REQUIRED_PAIR_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise EnergyBuildError(f"Pairs CSV missing required columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "pair_id": (row.get("pair_id") or "").strip(),
                    "idr_a": (row.get("idr_a") or "").strip(),
                    "idr_b": (row.get("idr_b") or "").strip(),
                }
            )
    if not rows:
        raise EnergyBuildError(f"No rows found in {path}")
    return rows


def find_unique_file(directory: Path, pattern: str, label: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise EnergyBuildError(
            f"Expected one {label} matching '{pattern}' in {directory}, found {len(matches)}."
        )
    return matches[0]


def discover_jobs(
    pair_dir: Path,
    job_dir_glob: str,
    pdb_pattern: str,
    dcd_pattern: str,
) -> list[JobInput]:
    jobs: list[JobInput] = []
    if not pair_dir.exists():
        return jobs

    for job_dir in sorted(path for path in pair_dir.glob(job_dir_glob) if path.is_dir()):
        pdb_file = find_unique_file(job_dir, pdb_pattern, "PDB file")
        dcd_file = find_unique_file(job_dir, dcd_pattern, "DCD file")
        if pdb_file is None or dcd_file is None:
            continue
        jobs.append(JobInput(job_id=job_dir.name, pdb_file=pdb_file, dcd_file=dcd_file))
    return jobs


def main() -> int:
    args = parse_args()
    try:
        pairs = read_pairs(args.pairs_csv)
        ensure_file(args.residues_file, "Residues CSV")
        params = pd.read_csv(args.residues_file).set_index("three")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        created = 0
        skipped: list[str] = []

        for row in pairs:
            pair_id = row["pair_id"]
            pair_dir = args.runs_dir / f"{args.pair_dir_prefix}{pair_id}" / args.temp_dir_name
            jobs = discover_jobs(
                pair_dir=pair_dir,
                job_dir_glob=args.job_dir_glob,
                pdb_pattern=args.pdb_pattern,
                dcd_pattern=args.dcd_pattern,
            )
            if len(jobs) < args.min_jobs:
                skipped.append(pair_id)
                continue

            ah_mats = []
            yu_mats = []
            pairs_array = None
            res1 = None
            res2 = None

            for job in jobs:
                u = mda.Universe(str(job.pdb_file), str(job.dcd_file))
                avg_ah_per_pair, avg_yu_per_pair, job_pairs_array, job_res1, job_res2 = calc_per_pair_average_energy(
                    u,
                    params,
                    eps_lj=args.eps_lj,
                    temp=args.temp,
                    ionic=args.ionic,
                    rc_lj=args.rc_lj,
                    rc_yu=args.rc_yu,
                )
                ah_mats.append(np.asarray(avg_ah_per_pair, dtype=float))
                yu_mats.append(np.asarray(avg_yu_per_pair, dtype=float))

                if pairs_array is None:
                    pairs_array = np.asarray(job_pairs_array)
                    res1 = np.asarray(job_res1)
                    res2 = np.asarray(job_res2)

            avg_ah_per_pair = np.mean(np.stack(ah_mats, axis=0), axis=0)
            avg_yu_per_pair = np.mean(np.stack(yu_mats, axis=0), axis=0)
            avg_total_per_pair = np.asarray(avg_ah_per_pair) + np.asarray(avg_yu_per_pair)

            payload = {
                "name": pair_id,
                "box_nm": args.box_nm,
                "temp": args.temp,
                "ionic": args.ionic,
                "n_jobs": len(jobs),
                "avg_ah_per_pair": np.asarray(avg_ah_per_pair),
                "avg_yu_per_pair": np.asarray(avg_yu_per_pair),
                "avg_total_per_pair": np.asarray(avg_total_per_pair),
                "pairs_array": np.asarray(pairs_array),
                "res1": np.asarray(res1),
                "res2": np.asarray(res2),
            }

            out_path = args.output_dir / f"energies_{pair_id}_{args.box_label}_sliced.pkl"
            with out_path.open("wb") as handle:
                pickle.dump(payload, handle)
            created += 1

        print(f"Wrote {created} energy pickle file(s) to {args.output_dir}")
        if skipped:
            print(
                f"Skipped {len(skipped)} pair(s) with fewer than {args.min_jobs} valid job folders.",
                file=sys.stderr,
            )
        return 0
    except EnergyBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Unexpected failure while creating energy PKLs: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
