#!/usr/bin/env bash
set -euo pipefail

# One-off helper: normalize CALVADOS filenames under run/ so they keep "__" in pair IDs.
# Not part of the pipeline.

RUN_DIR="${1:-run}"
BOX_DIR_NAME="${2:-298k_20.0nm}"

find "$RUN_DIR" -type d -name "test_*" | while IFS= read -r pair_dir; do
  pair_base="$(basename "$pair_dir")"              # test_<pair_id>
  pair_id="${pair_base#test_}"                     # <pair_id> with __
  sim_dir="$pair_dir/$BOX_DIR_NAME"
  [[ -d "$sim_dir" ]] || continue

  find "$sim_dir" -maxdepth 1 -type d -name "job_*" | while IFS= read -r job_dir; do
    job_name="$(basename "$job_dir")"              # job_0
    job_num="${job_name#job_}"                     # 0
    for ext in dcd log pdb xml out err; do
      find "$job_dir" -maxdepth 1 -type f -name "*.${ext}" | while IFS= read -r oldf; do
        oldbase="$(basename "$oldf")"
        # Skip already-correct names
        if [[ "$oldbase" == "${pair_id}"_*"_job_${job_num}."* ]] || [[ "$oldbase" == "${pair_id}_"*"_job_${job_num}."* ]]; then
          :
        fi
        case "$oldbase" in
          checkpoint.pdb|top.pdb|config.yaml|components.yaml|restart.chk|run.py|bonds.txt|job_*.sh) continue ;;
        esac
        suffix=""
        case "$oldbase" in
          *_298.out) suffix="_298.out" ;;
          *_298.err) suffix="_298.err" ;;
          *.dcd) suffix="_20.0nm_job_${job_num}.dcd" ;;
          *.log) suffix="_20.0nm_job_${job_num}.log" ;;
          *.pdb) suffix="_20.0nm_job_${job_num}.pdb" ;;
          *.xml) suffix="_20.0nm_job_${job_num}.xml" ;;
          *) continue ;;
        esac
        newf="$job_dir/${pair_id}${suffix}"
        if [[ "$oldf" != "$newf" ]]; then
          mv "$oldf" "$newf"
        fi
      done
    done
  done
done
