#!/usr/bin/env python3
"""Compute and save center-of-mass RDFs for CALVADOS pair trajectories.

This script reads a canonical pair table, locates trajectory files for each pair
under the user's real run layout, computes one center-of-mass RDF across all
available job folders, and saves both:
1. a per-pair pickle containing the mean RDF and replicate RDFs
2. a summary CSV listing the output pickle path and number of replicates used

Expected run layout:
run/test_<pair_id>/298k_20.0nm/job_0/
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np
import pandas as pd
from mdtraj import element

PAIR_COLUMNS = {"pair_id", "idr_a", "idr_b"}


class SaveRDFsError(Exception):
    pass


@dataclass
class ReplicateInput:
    replicate_id: str
    pdb_file: Path
    dcd_file: Path


@dataclass
class RDFResult:
    pair_id: str
    idr_a: str
    idr_b: str
    box_size_nm: float
    r_nm: np.ndarray
    rdf_mean: np.ndarray
    rdf_std: np.ndarray
    rdf_sem: np.ndarray
    replicate_ids: list[str]
    replicate_rdfs: dict[str, dict[str, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute COM RDFs for all pairs in a canonical pair table.")
    parser.add_argument("--pairs-csv", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--residues-file", required=True, type=Path,
                        help="Residues CSV containing at least 'one', 'MW', and 'sigmas' columns.")
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--combined-trajs-dir", type=Path, default=None)
    parser.add_argument("--pair-dir-prefix", default="test_")
    parser.add_argument("--temp-dir-name", default="298k_20.0nm")
    parser.add_argument("--job-dir-glob", default="job_*")
    parser.add_argument("--rdf-prefix", default="rdfs_")
    parser.add_argument("--rdf-box-label", default="20nm")
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--pdb-pattern", default="*.pdb")
    parser.add_argument("--dcd-pattern", default="*.dcd")
    parser.add_argument("--min-replicates", type=int, default=1)
    return parser.parse_args()


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SaveRDFsError(f"{label} not found: {path}")
    if not path.is_file():
        raise SaveRDFsError(f"{label} is not a file: {path}")


def read_pairs_table(path: Path) -> list[dict[str, str]]:
    ensure_file_exists(path, "Pairs CSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SaveRDFsError(f"Pairs CSV has no header: {path}")
        missing = PAIR_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise SaveRDFsError(f"Pairs CSV is missing required columns: {', '.join(sorted(missing))}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            pair_id = (row.get("pair_id") or "").strip()
            idr_a = (row.get("idr_a") or "").strip()
            idr_b = (row.get("idr_b") or "").strip()
            if not pair_id or not idr_a or not idr_b:
                raise SaveRDFsError(f"Incomplete row in pairs CSV at line {row_number}: {row}")
            rows.append({"pair_id": pair_id, "idr_a": idr_a, "idr_b": idr_b})
    if not rows:
        raise SaveRDFsError(f"No pair rows found in {path}.")
    return rows


def read_fasta_sequences(fasta_path: Path) -> dict[str, str]:
    ensure_file_exists(fasta_path, "FASTA file")
    sequences: dict[str, list[str]] = {}
    current_id = None
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                identifier = line[1:].strip().split()[0]
                if not identifier:
                    raise SaveRDFsError(f"Empty FASTA header encountered at line {line_number} in {fasta_path}.")
                if identifier in sequences:
                    raise SaveRDFsError(f"Duplicate FASTA identifier '{identifier}' found in {fasta_path}.")
                sequences[identifier] = []
                current_id = identifier
            else:
                if current_id is None:
                    raise SaveRDFsError(f"Sequence encountered before first FASTA header at line {line_number}.")
                sequences[current_id].append(line)
    final_sequences = {identifier: "".join(parts) for identifier, parts in sequences.items()}
    if not final_sequences:
        raise SaveRDFsError(f"No sequences found in FASTA file: {fasta_path}")
    return final_sequences


def read_residue_table(path: Path) -> pd.DataFrame:
    ensure_file_exists(path, "Residues file")
    table = pd.read_csv(path)
    required_columns = {"one", "MW", "sigmas"}
    missing = required_columns.difference(table.columns)
    if missing:
        raise SaveRDFsError(f"Residues file is missing required columns: {', '.join(sorted(missing))}")
    return table.set_index("one")


def find_unique_matching(directory: Path, pattern: str, required_substring: str, label: str) -> Path | None:
    matches = sorted([p for p in directory.glob(pattern) if required_substring in p.name])
    if not matches:
        return None
    if len(matches) > 1:
        raise SaveRDFsError(
            f"Expected one {label} containing '{required_substring}' matching '{pattern}' in {directory}, found {len(matches)}."
        )
    return matches[0]


def discover_replicates(pair_dir: Path, job_dir_glob: str, pdb_pattern: str, dcd_pattern: str) -> list[ReplicateInput]:
    replicate_inputs = []
    if not pair_dir.exists():
        return replicate_inputs
    for replicate_dir in sorted(path for path in pair_dir.glob(job_dir_glob) if path.is_dir()):
        job_suffix = replicate_dir.name.replace("job_", "job_")
        pdb_file = find_unique_matching(replicate_dir, pdb_pattern, job_suffix, "PDB file")
        dcd_file = find_unique_matching(replicate_dir, dcd_pattern, job_suffix, "DCD file")
        if pdb_file is None or dcd_file is None:
            # fallback to single-match behavior
            pdbs = sorted(replicate_dir.glob(pdb_pattern))
            dcds = sorted(replicate_dir.glob(dcd_pattern))
            pdb_file = pdbs[0] if len(pdbs) == 1 else None
            dcd_file = dcds[0] if len(dcds) == 1 else None
        if pdb_file is None or dcd_file is None:
            continue
        replicate_inputs.append(ReplicateInput(replicate_id=replicate_dir.name, pdb_file=pdb_file, dcd_file=dcd_file))
    return replicate_inputs


def build_pair_topology(idr_a: str, idr_b: str, sequences: dict[str, str], residues: pd.DataFrame) -> md.Topology:
    if idr_a not in sequences:
        raise SaveRDFsError(f"Sequence for '{idr_a}' not found in FASTA.")
    if idr_b not in sequences:
        raise SaveRDFsError(f"Sequence for '{idr_b}' not found in FASTA.")
    seq_a = sequences[idr_a]
    seq_b = sequences[idr_b]
    if not seq_a or not seq_b:
        raise SaveRDFsError(f"Empty sequence encountered for pair {idr_a} / {idr_b}.")

    top = md.Topology()
    chain_a = top.add_chain()
    residue_a = top.add_residue(idr_a, chain_a, resSeq=0)
    chain_b = top.add_chain()
    residue_b = top.add_residue(idr_b, chain_b, resSeq=1)

    def add_chain_atoms(sequence: str, residue_name: str, residue_obj: Any) -> None:
        for residue_code in sequence:
            if residue_code not in residues.index:
                raise SaveRDFsError(
                    f"Residue code '{residue_code}' from sequence '{residue_name}' is not present in the residues table indexed by the 'one' column."
                )
            mass = float(residues.loc[residue_code, "MW"])
            radius = float(residues.loc[residue_code, "sigmas"]) / 2.0
            element.Element._elements_by_symbol.pop(residue_code, None)
            pseudo_element = element.Element.__new__(element.Element, 1, residue_code, residue_code, mass, radius)
            top.add_atom(residue_code, element=pseudo_element, residue=residue_obj)

    add_chain_atoms(seq_a, idr_a, residue_a)
    add_chain_atoms(seq_b, idr_b, residue_b)
    for atom_index in range(chain_a.n_atoms - 1):
        top.add_bond(chain_a.atom(atom_index), chain_a.atom(atom_index + 1))
    for atom_index in range(chain_b.n_atoms - 1):
        top.add_bond(chain_b.atom(atom_index), chain_b.atom(atom_index + 1))
    return top


def compute_com_rdf_for_replicates(
    replicate_inputs: list[ReplicateInput],
    topology: md.Topology,
    box_size_nm: float,
    bin_width_nm: float,
    combined_trajs_dir: Path | None = None,
    pair_id: str | None = None,
    rdf_box_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    if not replicate_inputs:
        raise SaveRDFsError("No valid replicate inputs were provided for RDF calculation.")
    upper_nm = box_size_nm / 2.0
    combined_cm_trajectories = []
    per_replicate = {}

    for replicate in replicate_inputs:
        trajectory = md.load(str(replicate.dcd_file), top=topology)
        cm_topology = md.Topology()
        chain_1 = cm_topology.add_chain()
        residue_1 = cm_topology.add_residue("CM1", chain_1, resSeq=0)
        cm_topology.add_atom("CM1", element=trajectory.top.atom(0).element, residue=residue_1)
        chain_2 = cm_topology.add_chain()
        residue_2 = cm_topology.add_residue("CM2", chain_2, resSeq=1)
        cm_topology.add_atom("CM2", element=trajectory.top.atom(0).element, residue=residue_2)

        com_positions = [
            md.compute_center_of_mass(trajectory.atom_slice(trajectory.top.select("chainid 0"))),
            md.compute_center_of_mass(trajectory.atom_slice(trajectory.top.select("chainid 1"))),
        ]
        com_positions_array = np.swapaxes(np.array(com_positions), 0, 1)
        cm_traj = md.Trajectory(
            com_positions_array, cm_topology, trajectory.time, trajectory.unitcell_lengths, trajectory.unitcell_angles
        )
        combined_cm_trajectories.append(cm_traj)

        r_values, rdf_values = md.compute_rdf(
            cm_traj, [[0, 1]], r_range=(0, upper_nm), bin_width=bin_width_nm, periodic=True
        )
        per_replicate[replicate.replicate_id] = {"r_nm": np.array(r_values), "rdf": np.array(rdf_values)}

    joined = md.join(combined_cm_trajectories)
    if combined_trajs_dir is not None and pair_id is not None and rdf_box_label is not None:
        combined_trajs_dir.mkdir(parents=True, exist_ok=True)
        # save only DCD/PDB for convenience
        joined.save_dcd(str(combined_trajs_dir / f"{pair_id}_{rdf_box_label}.dcd"))
        joined[0].save_pdb(str(combined_trajs_dir / f"{pair_id}_{rdf_box_label}.pdb"))

    mean_r, mean_rdf = md.compute_rdf(joined, [[0, 1]], r_range=(0, upper_nm), bin_width=bin_width_nm, periodic=True)
    rdf_stack = np.vstack([per_replicate[replicate_id]["rdf"] for replicate_id in per_replicate])
    rdf_std = np.std(rdf_stack, axis=0, ddof=0)
    rdf_sem = rdf_std / np.sqrt(rdf_stack.shape[0])
    return np.array(mean_r), np.array(mean_rdf), rdf_std, rdf_sem, per_replicate


def save_rdf_pickle(result: RDFResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(result, handle)


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SaveRDFsError("No RDF summary rows were generated.")
    fieldnames = ["pair_id", "idr_a", "idr_b", "n_replicates", "box_size_nm", "bin_width_nm", "rdf_file"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        if args.bin_width <= 0:
            raise SaveRDFsError("--bin-width must be positive.")
        if args.box_size <= 0:
            raise SaveRDFsError("--box-size must be positive.")
        if args.min_replicates < 1:
            raise SaveRDFsError("--min-replicates must be at least 1.")

        pair_rows = read_pairs_table(args.pairs_csv)
        sequences = read_fasta_sequences(args.fasta)
        residues = read_residue_table(args.residues_file)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        skipped_pairs = []

        for pair_row in pair_rows:
            pair_id = pair_row["pair_id"]
            idr_a = pair_row["idr_a"]
            idr_b = pair_row["idr_b"]

            pair_dir = args.runs_dir / f"{args.pair_dir_prefix}{pair_id}" / args.temp_dir_name
            replicate_inputs = discover_replicates(
                pair_dir=pair_dir,
                job_dir_glob=args.job_dir_glob,
                pdb_pattern=args.pdb_pattern,
                dcd_pattern=args.dcd_pattern,
            )
            if len(replicate_inputs) < args.min_replicates:
                skipped_pairs.append(pair_id)
                continue

            topology = build_pair_topology(idr_a, idr_b, sequences, residues)
            r_nm, rdf_mean, rdf_std, rdf_sem, replicate_rdfs = compute_com_rdf_for_replicates(
                replicate_inputs=replicate_inputs,
                topology=topology,
                box_size_nm=args.box_size,
                bin_width_nm=args.bin_width,
                combined_trajs_dir=args.combined_trajs_dir,
                pair_id=pair_id,
                rdf_box_label=args.rdf_box_label,
            )
            rdf_result = {
                "pair_id": pair_id,
                "idr_a": idr_a,
                "idr_b": idr_b,
                "box_size_nm": args.box_size,
                "r_nm": r_nm,
                "rdf_mean": rdf_mean,
                "rdf_std": rdf_std,
                "rdf_sem": rdf_sem,
                "replicate_ids": [replicate.replicate_id for replicate in replicate_inputs],
                "replicate_rdfs": replicate_rdfs,
            }
            output_pickle = args.output_dir / f"{args.rdf_prefix}{pair_id}_{args.rdf_box_label}.pkl"
            save_rdf_pickle(rdf_result, output_pickle)

            summary_rows.append(
                {
                    "pair_id": pair_id,
                    "idr_a": idr_a,
                    "idr_b": idr_b,
                    "n_replicates": len(replicate_inputs),
                    "box_size_nm": args.box_size,
                    "bin_width_nm": args.bin_width,
                    "rdf_file": str(output_pickle),
                }
            )

        if not summary_rows:
            raise SaveRDFsError("No RDF files were written. Check that run folders contain matching PDB and DCD files.")
        write_summary_csv(summary_rows, args.summary_csv)
        print(f"Saved {len(summary_rows)} RDF pickle file(s) to {args.output_dir} and wrote summary CSV to {args.summary_csv}")
        if skipped_pairs:
            print(f"Skipped {len(skipped_pairs)} pair(s) with fewer than {args.min_replicates} valid replicates.", file=sys.stderr)
        return 0
    except SaveRDFsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: File system error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
