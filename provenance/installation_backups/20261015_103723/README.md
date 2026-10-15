# idr-interaction-analysis

A workflow for preparing pairwise CALVADOS simulations of intrinsically disordered regions (IDRs), extracting center-of-mass RDFs, computing corrected B22 values, and building sequence feature tables with optional B22 annotations.

## What this repository does

This repository covers the analysis workflow around pairwise IDR interactions:

1. generate ordered pair tables from IDR identifiers
2. prepare CALVADOS input folders
3. run CALVADOS externally
4. save RDFs from completed trajectories
5. compute corrected B22 values
6. compute sequence-derived pair features
7. optionally merge B22 values into the final feature table

The current workflow keeps pair order meaningful: `a` first and `b` second.

## Expected layout

```text
idr-interaction-analysis/
├── Snakefile
├── config/
│   └── config.yaml
├── inputs/
│   ├── all_tads_rggs_dis.fasta
│   ├── idra_list_example.txt
│   ├── idrb_list_example.txt
│   └── pairs_list_example.txt
├── resources/
│   └── residues.csv
├── scripts/
│   ├── generate_pairs.py
│   ├── prepare_calvados_inputs.py
│   ├── save_rdfs.py
│   ├── compute_b22.py
│   ├── compute_pair_features.py
│   └── plot_feature_vs_b22.py
├── run/
├── rdfs/
├── combined_trajs/
└── results/
    ├── prepared/
    ├── b22/
    ├── features/
    ├── analysis/
    └── figures/
```

## Installation

Create and activate your analysis environment, then install Snakemake and CALVADOS separately.

CALVADOS should be installed from the official repository:

```bash
git clone https://github.com/KULL-Centre/CALVADOS.git
cd CALVADOS
pip install -e .
```

Snakemake can be installed in the same environment:

```bash
pip install snakemake
```

## Core workflow

### 1. Generate ordered pairs

```bash
python scripts/generate_pairs.py   --fasta inputs/all_tads_rggs_dis.fasta   --pairs-file inputs/pairs_list_example.txt   --keep-order   --output results/prepared/pairs.csv
```

### 2. Prepare CALVADOS inputs

```bash
python scripts/prepare_calvados_inputs.py   --pairs-csv results/prepared/pairs.csv   --fasta inputs/all_tads_rggs_dis.fasta   --residues-file resources/residues.csv   --output-dir run
```

### 3. Run CALVADOS externally

This repository does not run CALVADOS itself. After input preparation, run CALVADOS in the generated job folders.

### 4. Save RDFs

```bash
python scripts/save_rdfs.py   --pairs-csv results/prepared/pairs.csv   --fasta inputs/all_tads_rggs_dis.fasta   --residues-file resources/residues.csv   --runs-dir run   --output-dir rdfs   --summary-csv rdfs/rdf_summary.csv   --combined-trajs-dir combined_trajs   --pair-dir-prefix test_   --temp-dir-name 298k_20.0nm   --job-dir-glob "job_*"   --rdf-prefix rdfs_   --rdf-box-label 20nm   --box-size 20.0   --bin-width 0.1   --pdb-pattern "*.pdb"   --dcd-pattern "*.dcd"   --min-replicates 1
```

### 5. Compute B22

```bash
python scripts/compute_b22.py   --rdf-summary-csv rdfs/rdf_summary.csv   --output-csv results/b22/b22_summary.csv   --integration-boundary 7.5   --apply-finite-size-correction
```

### 6. Compute pair features

Feature-only table:

```bash
python scripts/compute_pair_features.py   --pairs-csv results/prepared/pairs.csv   --fasta inputs/all_tads_rggs_dis.fasta   --residues-file resources/residues.csv   --output-csv results/features/pair_features.csv
```

Feature table with B22 columns:

```bash
python scripts/compute_pair_features.py   --pairs-csv results/prepared/pairs.csv   --fasta inputs/all_tads_rggs_dis.fasta   --residues-file resources/residues.csv   --b22-csv results/b22/b22_summary.csv   --output-csv results/analysis/pair_features_b22.csv
```

### 7. Plot feature-vs-B22 summaries

```bash
python scripts/plot_feature_vs_b22.py   --input-csv results/analysis/pair_features_b22.csv   --output-dir results/figures/feature_vs_b22
```

## Snakemake targets

Build everything declared in the current workflow:

```bash
snakemake --cores 1
```

Useful subsets:

```bash
snakemake --cores 1 core_inputs
snakemake --cores 1 post_simulation
snakemake --cores 1 feature_tables
```

## Main outputs

- `results/prepared/pairs.csv`
- `rdfs/rdf_summary.csv`
- `results/b22/b22_summary.csv`
- `results/features/pair_features.csv`
- `results/analysis/pair_features_b22.csv`

## Notes

- Pair order is preserved.
- Pair IDs use the `a__b` convention.
- The final feature table follows the legacy column naming style, with `a`/`b` replacing the earlier `rgg`/`tad` labels.
