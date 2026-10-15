# Existing-B22 convergence only: never depends on compute_b22 or RDF extraction.
def convergence_flag(value):
    if isinstance(value, bool): return value
    if str(value).lower() in ("true", "1", "yes"): return True
    if str(value).lower() in ("false", "0", "no"): return False
    raise ValueError("include_b22_convergence must be true or false")

CONVERGENCE_WITH_MAIN = convergence_flag(config.get("include_b22_convergence", False))
CONVERGENCE_REPORT = "results/b22_convergence/verified/verification_report.json"
CONVERGENCE_MAIN_INPUTS = [
    config["outputs"]["prepared_pairs_csv"],
    "results/prepared/calvados_inputs.done",
    config["outputs"]["rdf_summary_csv"],
    config["outputs"]["b22_csv"],
    config["outputs"]["features_csv"],
    config["outputs"]["features_with_b22_csv"],
    "results/figures/feature_vs_b22",
] if CONVERGENCE_WITH_MAIN else []

rule b22_convergence:
    input:
        CONVERGENCE_REPORT

rule verify_existing_b22_convergence:
    input:
        sources=expand("inputs/convergence_sampling/frozen_sources/Table_{sheet}.csv",
                       sheet=["S6C", "S6D", "S6E", "S7A", "S7B"]),
        configuration="config/convergence_sampling.json",
        script="scripts/verify_convergence_sampling_sources.py",
        main=CONVERGENCE_MAIN_INPUTS
    output:
        report=CONVERGENCE_REPORT,
        replica="results/b22_convergence/verified/replicate_count_convergence_summary.csv",
        trajectory="results/b22_convergence/verified/trajectory_final_change.csv",
        sampling="results/sampling/verified/B22_distribution_stability_by_stage.csv"
    shell:
        "python {input.script:q} --config {input.configuration:q}"
