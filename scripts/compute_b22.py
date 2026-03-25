#!/usr/bin/env python3
"""Compute corrected B22 values from saved RDF pickle files using helper functions."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdf_b22_functions import corrected_rdfs_b22

REQUIRED_COLUMNS = {
    "pair_id",
    "idr_a",
    "idr_b",
    "n_replicates",
    "box_size_nm",
    "bin_width_nm",
    "rdf_file",
}


class ComputeB22Error(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute corrected B22 values from RDF pickle files.")
    parser.add_argument("--rdf-summary-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--apply-finite-size-correction", action="store_true",
                        help="Retained for CLI compatibility; helper applies correction.")
    parser.add_argument(
        "--integration-boundary",
        type=float,
        default=7.5,
        help="Upper integration boundary in nm passed to corrected_rdfs_b22. Default: 7.5.",
    )
    return parser.parse_args()


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise ComputeB22Error(f"{label} not found: {path}")
    if not path.is_file():
        raise ComputeB22Error(f"{label} is not a file: {path}")


def read_rdf_summary(path: Path) -> list[dict[str, str]]:
    ensure_file_exists(path, "RDF summary CSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ComputeB22Error(f"RDF summary CSV has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ComputeB22Error(f"RDF summary CSV is missing required columns: {', '.join(sorted(missing))}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            clean_row = {(key or ""): (value or "").strip() for key, value in row.items()}
            if not clean_row["pair_id"]:
                raise ComputeB22Error(f"Missing pair_id in RDF summary CSV at line {row_number}.")
            rows.append(clean_row)
    if not rows:
        raise ComputeB22Error(f"No rows found in RDF summary CSV: {path}")
    return rows


def load_rdf_pickle(path: Path) -> dict[str, Any]:
    ensure_file_exists(path, "RDF pickle")
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if hasattr(loaded, "__dict__"):
        loaded_dict = dict(vars(loaded))
    elif isinstance(loaded, dict):
        loaded_dict = loaded
    else:
        raise ComputeB22Error(f"Unsupported RDF pickle type in {path}: {type(loaded).__name__}")
    required_keys = {"r_nm", "rdf_mean", "box_size_nm"}
    missing = required_keys.difference(loaded_dict)
    if missing:
        raise ComputeB22Error(f"RDF pickle {path} is missing required key(s): {', '.join(sorted(missing))}")
    return loaded_dict


def main() -> int:
    args = parse_args()
    try:
        summary_rows = read_rdf_summary(args.rdf_summary_csv)
        output_rows: list[dict[str, Any]] = []

        for row in summary_rows:
            rdf_path = Path(row["rdf_file"])
            if not rdf_path.is_absolute():
                candidate_paths = [
                    rdf_path,
                    Path.cwd() / rdf_path,
                    args.rdf_summary_csv.parent / rdf_path,
                ]
                resolved_path = None
                for candidate in candidate_paths:
                    if candidate.exists():
                        resolved_path = candidate.resolve()
                        break
                if resolved_path is None:
                    raise ComputeB22Error(
                        f"RDF pickle not found using any expected relative path for: {row['rdf_file']}"
                    )
                rdf_path = resolved_path
            rdf_payload = load_rdf_pickle(rdf_path)
            r_nm = np.asarray(rdf_payload["r_nm"], dtype=float)
            rdf_mean = np.asarray(rdf_payload["rdf_mean"], dtype=float)
            box_size_nm = float(rdf_payload["box_size_nm"])

            rdf_df = pd.DataFrame({"run_1": [r_nm, rdf_mean]})
            corrected_result = corrected_rdfs_b22(
                rdf_df,
                args.integration_boundary,
                box_size=box_size_nm,
            )
            if not isinstance(corrected_result, (tuple, list)) or len(corrected_result) < 2:
                raise ComputeB22Error(
                    "corrected_rdfs_b22 returned an unexpected result; expected at least "
                    "(rdfs_dict, b22s_dict)."
                )
            rdfs_dict = corrected_result[0]
            b22s_dict = corrected_result[1]

            if not isinstance(b22s_dict, dict):
                raise ComputeB22Error("corrected_rdfs_b22 did not return a B22 dictionary.")

            hummer = np.nan
            ganguly = np.nan
            corr1_log10 = np.nan
            corr2_log10 = np.nan
            corr2_log10_no_pos = np.nan
            raw_b22 = np.nan

            run_key = "run_1"
            if run_key in b22s_dict:
                run_b22 = b22s_dict[run_key]
                raw_b22 = run_b22.get("b22", np.nan)
                hummer = run_b22.get("b22_corr1", np.nan)
                ganguly = run_b22.get("b22_corr2", np.nan)

                if np.isfinite(hummer) and hummer < 0:
                    corr1_log10 = np.log10(-hummer)
                if np.isfinite(ganguly) and ganguly < 0:
                    corr2_log10 = np.log10(-ganguly)
                if np.isfinite(ganguly) and ganguly < 0:
                    corr2_log10_no_pos = np.log10(-ganguly)

            output_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "idr_a": row["idr_a"],
                    "idr_b": row["idr_b"],
                    "b22_mean": f"{float(hummer):.10g}" if np.isfinite(hummer) else "",
                    "B22": f"{float(raw_b22):.10g}" if np.isfinite(raw_b22) else "",
                    "Corrected_B22_Hummer": f"{float(hummer):.10g}" if np.isfinite(hummer) else "",
                    "Corrected_B22_Ganguly": f"{float(ganguly):.10g}" if np.isfinite(ganguly) else "",
                    "B22_corr1_log10": f"{float(corr1_log10):.10g}" if np.isfinite(corr1_log10) else "",
                    "B22_corr2_log10": f"{float(corr2_log10):.10g}" if np.isfinite(corr2_log10) else "",
                    "B22_corr2_log10_no_pos": f"{float(corr2_log10_no_pos):.10g}" if np.isfinite(corr2_log10_no_pos) else "",
                    "n_replicates": row["n_replicates"],
                    "rdf_file": row["rdf_file"],
                }
            )

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "pair_id", "idr_a", "idr_b", "b22_mean", "B22", "Corrected_B22_Hummer",
                    "Corrected_B22_Ganguly", "B22_corr1_log10", "B22_corr2_log10",
                    "B22_corr2_log10_no_pos", "n_replicates", "rdf_file",
                ],
            )
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"Wrote B22 summary for {len(output_rows)} pair(s) to {args.output_csv}")
        return 0
    except ComputeB22Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: File system error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Unexpected failure while computing B22 values: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
