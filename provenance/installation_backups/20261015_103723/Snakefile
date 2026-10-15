configfile: "config/config.yaml"


rule all:
    input:
        config["outputs"]["prepared_pairs_csv"],
        "results/prepared/calvados_inputs.done",
        config["outputs"]["rdf_summary_csv"],
        config["outputs"]["b22_csv"],
        config["outputs"]["features_csv"],
        config["outputs"]["features_with_b22_csv"], 
        "results/figures/feature_vs_b22"

rule core_inputs:
    input:
        config["outputs"]["prepared_pairs_csv"],
        "results/prepared/calvados_inputs.done"

rule post_simulation:
    input:
        config["outputs"]["rdf_summary_csv"],
        config["outputs"]["b22_csv"]

rule feature_tables:
    input:
        config["outputs"]["features_csv"],
        config["outputs"]["features_with_b22_csv"]

rule generate_pairs:
    input:
        fasta=config["input_fasta"]
    output:
        csv=config["outputs"]["prepared_pairs_csv"]
    params:
        mode=config["pairing"]["mode"],
        pairs_file=config["pairing"]["pairs_file"],
        list_a=config["pairing"]["list_a"],
        list_b=config["pairing"]["list_b"],
        allow_self="--allow-self-pairs" if config["pairing"]["allow_self_pairs"] else "",
        keep_order="--keep-order" if config["pairing"].get("keep_order", False) else ""
    shell:
        """
        if [ "{params.mode}" = "explicit_pairs" ]; then \
          python scripts/generate_pairs.py \
            --fasta {input.fasta} \
            --pairs-file {params.pairs_file} \
            {params.keep_order} \
            --output {output.csv}; \
        elif [ "{params.mode}" = "two_lists" ]; then \
          python scripts/generate_pairs.py \
            --fasta {input.fasta} \
            --list-a {params.list_a} \
            --list-b {params.list_b} \
            {params.allow_self} \
            {params.keep_order} \
            --output {output.csv}; \
        else \
          echo "Unsupported pairing mode: {params.mode}" >&2; \
          exit 1; \
        fi
        """

rule prepare_calvados_inputs:
    input:
        pairs_csv=config["outputs"]["prepared_pairs_csv"],
        fasta=config["input_fasta"],
        residues=config["resources"]["residues_file"]
    output:
        touch("results/prepared/calvados_inputs.done")
    params:
        runs_dir=config["paths"]["runs_dir"],
        replicates=config["calvados"]["replicates"],
        box=config["calvados"]["box"],
        temp=config["calvados"]["temp"],
        ionic=config["calvados"]["ionic"],
        ph=config["calvados"]["ph"],
        steps=config["calvados"]["steps"],
        wfreq=config["calvados"]["wfreq"],
        platform=config["calvados"]["platform"],
        threads=config["calvados"]["threads"],
        restart_mode=config["calvados"]["restart_mode"],
        restart_file=config["calvados"]["restart_file"],
        topol=config["calvados"]["topol"],
        runtime=config["calvados"]["runtime"],
        charge_termini=config["calvados"]["charge_termini"],
        write_job_script="--write-job-script" if config["calvados"]["write_job_script"] else "",
        job_script_name=config["calvados"]["job_script_name"]
    shell:
        """
        python scripts/prepare_calvados_inputs.py \
          --pairs-csv {input.pairs_csv} \
          --fasta {input.fasta} \
          --residues-file {input.residues} \
          --output-dir {params.runs_dir} \
          --replicates {params.replicates} \
          --box {params.box} \
          --temp {params.temp} \
          --ionic {params.ionic} \
          --ph {params.ph} \
          --steps {params.steps} \
          --wfreq {params.wfreq} \
          --platform {params.platform} \
          --threads {params.threads} \
          --restart-mode {params.restart_mode} \
          --restart-file {params.restart_file} \
          --topol {params.topol} \
          --runtime {params.runtime} \
          --charge-termini {params.charge_termini} \
          {params.write_job_script} \
          --job-script-name {params.job_script_name}
        && touch {output}
        """

rule save_rdfs:
    input:
        pairs_csv=config["outputs"]["prepared_pairs_csv"],
        fasta=config["input_fasta"],
        residues=config["resources"]["residues_file"]
    output:
        csv=config["outputs"]["rdf_summary_csv"]
    params:
        runs_dir=config["paths"]["runs_dir"],
        output_dir=config["paths"]["rdfs_dir"],
        combined_trajs_dir=config["paths"]["combined_trajs_dir"],
        pair_dir_prefix=config["naming"]["pair_dir_prefix"],
        temp_dir_name=config["naming"]["temp_dir_name"],
        job_dir_glob=config["naming"]["job_dir_glob"],
        rdf_prefix=config["naming"]["rdf_prefix"],
        rdf_box_label=config["naming"]["rdf_box_label"],
        box_size=config["rdf"]["box_size"],
        bin_width=config["rdf"]["bin_width"],
        pdb_pattern=config["rdf"]["pdb_pattern"],
        dcd_pattern=config["rdf"]["dcd_pattern"],
        min_replicates=config["rdf"]["min_replicates"]
    shell:
        """
        python scripts/save_rdfs.py \
          --pairs-csv {input.pairs_csv} \
          --fasta {input.fasta} \
          --residues-file {input.residues} \
          --runs-dir {params.runs_dir} \
          --output-dir {params.output_dir} \
          --summary-csv {output.csv} \
          --combined-trajs-dir {params.combined_trajs_dir} \
          --pair-dir-prefix {params.pair_dir_prefix} \
          --temp-dir-name {params.temp_dir_name} \
          --job-dir-glob '{params.job_dir_glob}' \
          --rdf-prefix {params.rdf_prefix} \
          --rdf-box-label {params.rdf_box_label} \
          --box-size {params.box_size} \
          --bin-width {params.bin_width} \
          --pdb-pattern '{params.pdb_pattern}' \
          --dcd-pattern '{params.dcd_pattern}' \
          --min-replicates {params.min_replicates}
        """

rule compute_b22:
    input:
        config["outputs"]["rdf_summary_csv"]
    output:
        config["outputs"]["b22_csv"]
    params:
        finite_size="--apply-finite-size-correction" if config["b22"]["apply_finite_size_correction"] else "",
        integration_boundary=config["b22"]["integration_boundary"]
    shell:
        """
        python scripts/compute_b22.py \
          --rdf-summary-csv {input} \
          --output-csv {output} \
          --integration-boundary {params.integration_boundary} \
          {params.finite_size}
        """

rule compute_pair_features:
    input:
        pairs_csv=config["outputs"]["prepared_pairs_csv"],
        fasta=config["input_fasta"],
        residues=config["resources"]["residues_file"]
    output:
        config["outputs"]["features_csv"]
    shell:
        """
        python scripts/compute_pair_features.py \
          --pairs-csv {input.pairs_csv} \
          --fasta {input.fasta} \
          --residues-file {input.residues} \
          --output-csv {output}
        """

rule compute_pair_features_with_b22:
    input:
        pairs_csv=config["outputs"]["prepared_pairs_csv"],
        fasta=config["input_fasta"],
        residues=config["resources"]["residues_file"],
        b22=config["outputs"]["b22_csv"]
    output:
        config["outputs"]["features_with_b22_csv"]
    shell:
        """
        python scripts/compute_pair_features.py \
          --pairs-csv {input.pairs_csv} \
          --fasta {input.fasta} \
          --residues-file {input.residues} \
          --b22-csv {input.b22} \
          --output-csv {output}
        """

rule plots:
    input:
        "results/figures/feature_vs_b22/Opposite_Charge_Number_vs_B22_corr1_log10.png",
        "results/figures/feature_vs_b22/Normalized_OCN_vs_B22_corr1_log10.png",
        "results/figures/feature_vs_b22/Charge_Complementarity_vs_B22_corr1_log10.png",
        "results/figures/feature_vs_b22/Average_lambda_vs_B22_corr1_log10.png"

rule plot_feature_vs_b22:
    input:
        csv=config["outputs"]["features_with_b22_csv"]
    output:
        directory("results/figures/feature_vs_b22")
    shell:
        """
        python scripts/plot_feature_vs_b22.py \
          --input-csv {input.csv} \
          --output-dir {output}
        """

rule supp4:
    output:
        directory("sims_supplementary")
    shell:
        """
        python scripts/supp4.py
        """

rule fig4_manual:
    input:
        table=config["outputs"]["features_with_b22_csv"]
    output:
        directory("results/figures/fig4")
    params:
        rdf_strong="rdfs/rdfs_Q14152_5_3__P98177_463_504_20nm.pkl",
        rdf_weak="rdfs/rdfs_Q14152_5_3__P35716_353_367_20nm.pkl",
        energy_dir="test_energies",
        aa_script="scripts/amino_acids_energy_interactions.py",
        box_nm=20,
        rdf_last=7.5,
        rdf_delta=2.0
    shell:
        """
        python scripts/fig4.py \
          --table {input.table} \
          --outdir {output} \
          --rdf_strong {params.rdf_strong} \
          --rdf_weak {params.rdf_weak} \
          --energy_dir {params.energy_dir} \
          --aa_script {params.aa_script} \
          --box_nm {params.box_nm} \
          --rdf_last {params.rdf_last} \
          --rdf_delta {params.rdf_delta}
        """

rule compute_pair_energies:
    input:
        pairs_csv=config["outputs"]["prepared_pairs_csv"],
        residues=config["resources"]["residues_file"]
    output:
        directory("test_energies")
    params:
        runs_dir=config["paths"]["runs_dir"],
        pair_dir_prefix=config["naming"]["pair_dir_prefix"],
        temp_dir_name=config["naming"]["temp_dir_name"],
        job_dir_glob=config["naming"]["job_dir_glob"],
        box_label=config["naming"]["rdf_box_label"],
        box_nm=config["rdf"]["box_size"],
        temp=config["calvados"]["temp"],
        ionic=config["calvados"]["ionic"],
        pdb_pattern="*_job_*.pdb",
        dcd_pattern=config["rdf"]["dcd_pattern"]
    shell:
        """
        python scripts/compute_pair_energies.py \
          --pairs-csv {input.pairs_csv} \
          --runs-dir {params.runs_dir} \
          --residues-file {input.residues} \
          --output-dir {output} \
          --pair-dir-prefix {params.pair_dir_prefix} \
          --temp-dir-name {params.temp_dir_name} \
          --job-dir-glob '{params.job_dir_glob}' \
          --pdb-pattern '{params.pdb_pattern}' \
          --dcd-pattern '{params.dcd_pattern}' \
          --box-label {params.box_label} \
          --box-nm {params.box_nm} \
          --temp {params.temp} \
          --ionic {params.ionic}
        """
