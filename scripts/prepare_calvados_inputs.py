#!/usr/bin/env python3
"""Prepare CALVADOS input folders from a canonical pair table using the user's preferred run layout."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from calvados.cfg import Components, Config

REQUIRED_COLUMNS = {"pair_id", "idr_a", "idr_b"}


class PrepareInputsError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CALVADOS input folders for a set of IDR pairs.")
    parser.add_argument("--pairs-csv", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--residues-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--box", type=float, default=20.0)
    parser.add_argument("--temp", type=float, default=298.0)
    parser.add_argument("--ionic", type=float, default=0.15)
    parser.add_argument("--ph", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=100_000_000)
    parser.add_argument("--wfreq", type=int, default=10000)
    parser.add_argument("--platform", default="CPU")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--restart-mode", default="checkpoint")
    parser.add_argument("--restart-file", default="restart.chk")
    parser.add_argument("--topol", default="grid")
    parser.add_argument("--runtime", type=int, default=0)
    parser.add_argument("--charge-termini", choices=["none", "N", "C", "both"], default="both")
    parser.add_argument("--write-job-script", action="store_true")
    parser.add_argument("--job-script-name", default="run_calvados.sh")
    parser.add_argument("--pair-dir-prefix", default="test_")
    return parser.parse_args()


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise PrepareInputsError(f"{label} not found: {path}")
    if not path.is_file():
        raise PrepareInputsError(f"{label} is not a file: {path}")


def read_pairs_table(path: Path) -> list[dict[str, str]]:
    ensure_file_exists(path, "Pairs CSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PrepareInputsError(f"Pairs CSV has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise PrepareInputsError(f"Pairs CSV is missing required column(s): {', '.join(sorted(missing))}")
        rows = []
        seen_pair_ids = set()
        for row_number, row in enumerate(reader, start=2):
            pair_id = (row.get("pair_id") or "").strip()
            idr_a = (row.get("idr_a") or "").strip()
            idr_b = (row.get("idr_b") or "").strip()
            if not pair_id or not idr_a or not idr_b:
                raise PrepareInputsError(f"Incomplete row in {path} at line {row_number}: {row}")
            if pair_id in seen_pair_ids:
                raise PrepareInputsError(f"Duplicate pair_id '{pair_id}' found in {path}.")
            seen_pair_ids.add(pair_id)
            rows.append({"pair_id": pair_id, "idr_a": idr_a, "idr_b": idr_b})
    if not rows:
        raise PrepareInputsError(f"No pair rows found in {path}.")
    return rows


def build_base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sysname": "placeholder",
        "box": [args.box, args.box, args.box],
        "temp": args.temp,
        "ionic": args.ionic,
        "pH": args.ph,
        "topol": args.topol,
        "wfreq": args.wfreq,
        "steps": args.steps,
        "threads": args.threads,
        "runtime": args.runtime,
        "platform": args.platform,
        "restart": args.restart_mode,
        "frestart": args.restart_file,
        "verbose": True,
    }


def make_run_directory(base_output_dir: Path, pair_id: str, box: float, temp: float, replicate_index: int, pair_dir_prefix: str) -> Path:
    pair_root = base_output_dir / f"{pair_dir_prefix}{pair_id}" / f"{int(round(temp))}k_{box:.1f}nm"
    run_dir = pair_root / f"job_{replicate_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_job_script(run_dir: Path, script_name: str) -> None:
    script_path = run_dir / script_name
    script_text = "#!/usr/bin/env bash\nset -euo pipefail\npython run.py --path .\n"
    script_path.write_text(script_text, encoding="utf-8")
    current_mode = script_path.stat().st_mode
    script_path.chmod(current_mode | 0o111)


def write_calvados_files_for_pair(*, pair_row: dict[str, str], args: argparse.Namespace, base_config: dict[str, Any]) -> int:
    pair_id = pair_row["pair_id"]
    idr_a = pair_row["idr_a"]
    idr_b = pair_row["idr_b"]

    created = 0
    for replicate_index in range(args.replicates):
        run_dir = make_run_directory(args.output_dir, pair_id, args.box, args.temp, replicate_index, args.pair_dir_prefix)
        sysname = f"{pair_id}_{args.box:.1f}nm_job_{replicate_index}"

        config = Config(**base_config)
        config.config["sysname"] = sysname
        config.write(str(run_dir), name="config.yaml")

        components = Components(
            molecule_type="protein",
            nmol=1,
            restraint=False,
            charge_termini=args.charge_termini,
            fresidues=str(args.residues_file.resolve()),
            ffasta=str(args.fasta.resolve()),
        )
        components.add(name=idr_a, nmol=1, charge_termini=args.charge_termini)
        components.add(name=idr_b, nmol=1, charge_termini=args.charge_termini)
        components.write(str(run_dir), name="components.yaml")

        if args.write_job_script:
            write_job_script(run_dir, args.job_script_name)
        created += 1
    return created


def main() -> int:
    args = parse_args()
    try:
        ensure_file_exists(args.fasta, "FASTA file")
        ensure_file_exists(args.residues_file, "Residues file")
        if args.replicates < 1:
            raise PrepareInputsError("--replicates must be at least 1.")
        if args.box <= 0:
            raise PrepareInputsError("--box must be positive.")
        if args.temp <= 0:
            raise PrepareInputsError("--temp must be positive.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pair_rows = read_pairs_table(args.pairs_csv)
        base_config = build_base_config(args)

        total_run_dirs = 0
        for pair_row in pair_rows:
            total_run_dirs += write_calvados_files_for_pair(pair_row=pair_row, args=args, base_config=base_config)

        print(f"Prepared {total_run_dirs} CALVADOS run folder(s) for {len(pair_rows)} pair(s) in {args.output_dir}")
        return 0
    except PrepareInputsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: Failed while writing output files: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
