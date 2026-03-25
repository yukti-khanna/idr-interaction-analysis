#!/usr/bin/env python3
# file: make_fig5_panels.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# Nature-style rcParams
# =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# =========================
# Constants / I/O
# =========================
OUTPUT_DIR = Path("sims_supplementary")
DATA_DIR = OUTPUT_DIR / "data"
FIGSIZE = (1.8, 1.15)
DPI_PNG = 600

COLOR_BLUE = "#0072B2"   # Okabe–Ito blue
COLOR_GREY = "#646161"      # visible grey

RNG_SEED = 12345
N_BINS = 30
MAX_BG_POINTS_F = 100_000  # non-RBP background downsampling in Panel F

# Default input files (can be overridden from the command line)
TRAINING_CSV = "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
HELDOUT_CSV = "taken_out_preds_with_hybrid_021125_sel_cols.csv"
UNSIM_CSV = "unsim_preds_with_hybrid_021125_sel_cols.csv"
PAIRS25_CSV = "results/analysis/pair_features_b22.csv"
INITIAL_SET_CSV = "results/analysis/pair_features_b22.csv"
FEATURE_IMPORTANCE_CSV = "feature_pruning_results/permutation_importance_test.csv"
RBP_UNIID_FILE = "rg_rbp_unids.txt"

# Axis labels (Title Case + correct math)
LBL_X_OCN   = "Opposite Charge Number"
LBL_X_AL   = "Average λ"
LBL_X_ANCPR   = "Average NCPR"
LBL_X_CC   = "Charge Complementarity"
LBL_X_SIM   = r"$log_{10}(-B_{22})$"
LBL_Y_SIM   = r"$log_{10}(-B_{22})$"
LBL_Y_PRED  = r"Predicted $y=\log_{10}(-B_{22})$"
LBL_Y_PRED_HYB = r"Hybrid Predicted $y=\log_{10}(-B_{22})$"

# Panel limit lines (requested thresholds)
LAMBDA_INTERMEDIATE_BAND = (0.45, 0.56)
NCPR_NEUTRAL_BAND = (-0.05, 0.05)
CHARGE_COMPLEMENTARITY_MIN = 0.0
LIMIT_LINE_STYLE = dict(color="red", lw=0.8, ls="--")

# Canonical feature list (Panel B filter; OK to use λ for CALVADOS stickiness)
DEFAULT_FEATURES = [
    "Average_lambda","Min_lambda","Average_sigma","Average_charge","Average_NCPR",
    "Charge_Complementarity","Total_Length","Average_Interaction","Average_SCD","Average_SHD",
    "Opposite_Charge_Number","RGG_length","charge_RGG","lambda_RGG","NCPR_RGG","SCD_RGG","SHD_RGG",
    "TAD_length","charge_TAD","lambda_TAD","NCPR_TAD","SCD_TAD","SHD_TAD",
    "Normalized_OCN","Charge_Asymmetry","Hydropathy_Diff","Lambda_Ratio",
]

# Pretty feature labels for Panel B (Title Case; λ reserved for stickiness features)
PRETTY_FEATURE_LABELS = {
    "Average_lambda": "Average λ",
    "Min_lambda": "Min λ",
    "Average_sigma": "Average σ",
    "Average_charge": "Average Charge",
    "Average_NCPR": "Average NCPR",
    "Charge_Complementarity": "Charge Complementarity",
    "Total_Length": "Total Length",
    "Average_Interaction": "Average Interaction",
    "Average_SCD": "Average SCD",
    "Average_SHD": "Average SHD",
    "Opposite_Charge_Number": "Opposite-Charge Number",
    "RGG_length": "RGG Length",
    "charge_RGG": "Charge (RGG)",
    "lambda_RGG": "λ (RGG)",
    "NCPR_RGG": "NCPR (RGG)",
    "SCD_RGG": "SCD (RGG)",
    "SHD_RGG": "SHD (RGG)",
    "TAD_length": "TAD Length",
    "charge_TAD": "Charge (TAD)",
    "lambda_TAD": "λ (TAD)",
    "NCPR_TAD": "NCPR (TAD)",
    "SCD_TAD": "SCD (TAD)",
    "SHD_TAD": "SHD (TAD)",
    "Normalized_OCN": "Normalized OCN",
    "Charge_Asymmetry": "Charge Asymmetry",
    "Hydropathy_Diff": "Hydropathy Difference",
    "Lambda_Ratio": "λ Ratio",
}

# =========================
# Utilities
# =========================
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, low_memory=False)

def _read_feature_importance_table(path: str | Path) -> pd.DataFrame:
    """
    Robustly read permutation importance from:
      - CSV with headers, or
      - Series-saved CSV ['Unnamed: 0','0'], or
      - Headerless 2-col TSV/CSV (feature<sep>importance).
    Returns ['feature','importance'].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {path}")

    df1 = pd.read_csv(path, low_memory=False)
    cols = [str(c) for c in df1.columns]

    cand_feat = {"feature", "Feature", "variable", "var", "name", "Name"}
    cand_imp  = {"importance","Importance","weight","score","permutation_importance",
                 "importance_mean","perm_importance"}
    lc_map = {c.lower(): c for c in cols}
    fcol = next((lc_map[c] for c in cand_feat if c.lower() in lc_map), None)
    icol = next((lc_map[c] for c in cand_imp  if c.lower() in lc_map), None)
    if fcol and icol:
        out = df1[[fcol, icol]].copy()
        out.columns = ["feature", "importance"]
        return out

    if df1.shape[1] == 2:
        a, b = df1.columns.tolist()
        if ("unnamed" in str(a).lower()) or ("index" in str(a).lower()):
            return df1.rename(columns={a: "feature", b: "importance"})[["feature","importance"]]
        if ("unnamed" in str(b).lower()) or ("index" in str(b).lower()):
            return df1.rename(columns={b: "feature", a: "importance"})[["feature","importance"]]

    df2 = pd.read_csv(path, sep=None, engine="python", header=None, names=["feature","importance"])
    return df2[["feature","importance"]]


def _normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    return {c.lower(): c for c in df.columns}

def find_col(df: pd.DataFrame, candidates: Iterable[str], what: str) -> str:
    """Case-insensitive column autodetect with explicit error."""
    lut = _normalize_columns(df)
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in lut:
            return lut[lc]
    raise KeyError(
        f"Could not detect column for '{what}'. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )

def get_pair_col(df: pd.DataFrame) -> str:
    candidates = ["pair", "pair_id", "Pair", "pair_name", "Pair_ID", "name"]
    return find_col(df, candidates, "pair identifier")



def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """Pearson r with finite-mask and zero-variance guards."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 2:
        return (np.nan, n)
    xs = x[m]; ys = y[m]
    if np.allclose(np.std(xs), 0) or np.allclose(np.std(ys), 0):
        return (np.nan, n)
    r = np.corrcoef(xs, ys)[0, 1]
    return float(r), n

def binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Kept for CSV outputs; not plotted in E/F (we plot linear fits)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    xs = x[m]; ys = y[m]
    if xs.size == 0:
        return pd.DataFrame(columns=["bin_center", "median", "n"])
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return pd.DataFrame(columns=["bin_center", "median", "n"])
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    inds = np.digitize(xs, edges, right=False) - 1
    rows = []
    for i in range(n_bins):
        sel = inds == i
        if np.any(sel):
            rows.append({"bin_center": centers[i], "median": float(np.median(ys[sel])), "n": int(np.sum(sel))})
    return pd.DataFrame(rows, columns=["bin_center", "median", "n"])

def linear_fit_params(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares slope, intercept on finite points; NaN-safe."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return (np.nan, np.nan)
    xs, ys = x[m], y[m]
    if np.allclose(np.std(xs), 0) or np.allclose(np.std(ys), 0):
        return (np.nan, np.nan)
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)

def identity_limits(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Shared min/max with small padding for identity plots."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return (0.0, 1.0)
    data = np.concatenate([x[m], y[m]])
    dmin, dmax = float(np.min(data)), float(np.max(data))
    if dmin == dmax:
        dmin -= 0.5; dmax += 0.5
    pad = 0.05 * (dmax - dmin)
    return dmin - pad, dmax + pad

def save_fig(fig: mpl.figure.Figure, name: str) -> None:
    """Save both PDF (vector) and PNG (dpi=600), tight border."""
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

import matplotlib.patches as mpatches

def save_fig_with_right_gutter(fig, ax, name, right_gutter_in=0.12):
    # right_gutter_in is extra white space on the RIGHT in inches
    # Create a zero-width rectangle just outside the right edge of the axes.
    w = fig.get_figwidth()
    gutter_frac = right_gutter_in / w  # convert inches -> axes fraction
    extra = mpatches.Rectangle((1.0, 0.0), gutter_frac, 1.0,
                               transform=ax.transAxes, fill=False, linewidth=0)
    pdf = OUTPUT_DIR / f"{name}.pdf"
    png = OUTPUT_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", bbox_extra_artists=[extra], pad_inches=0.01)
    fig.savefig(png, dpi=DPI_PNG, bbox_inches="tight", bbox_extra_artists=[extra], pad_inches=0.01)
    plt.close(fig)

def prettify_feature_label(raw: str) -> str:
    """Curated pretty labels; fallback to Title Case."""
    if not isinstance(raw, str):
        return str(raw)
    if raw in PRETTY_FEATURE_LABELS:
        return PRETTY_FEATURE_LABELS[raw]
    return raw.replace("_", " ").title()




# =========================
# Panel E — Simulated Scaling: Normalized OCN vs Simulated y (Training + Held-Out)
# =========================
def panel_A():
    df_tr = _read_csv(PAIRS25_CSV)


    y_sim_tr = find_col(df_tr, ["B22_corr1_log10", "y_sim", "log10_negB22_sim"], "simulated y (Test-Set)")
    ocn_tr = find_col(df_tr, ["Average_lambda", "Avg_lambda", "lambda_avg"], "Average λ (Test-Set)")
    pair_tr = get_pair_col(df_tr)


    # Save points
    pts_tr = df_tr[[pair_tr, ocn_tr, y_sim_tr]].copy()
    pts_tr.insert(0, "dataset", "Test")
    pts_tr.columns = ["dataset", "pair", "avg_lambda", "y_sim"]

  
    pts = pd.concat([pts_tr,], ignore_index=True)
    pts.to_csv(DATA_DIR / "avg_lambda_test_set_25_points.csv", index=False)

    # Also keep binned CSV (compat), but do not plot it
    med_tr = binned_median(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy(), n_bins=N_BINS)
    med_tr.insert(0, "dataset", "Test")
    # Linear fits (plotted)
    m_tr, b_tr = linear_fit_params(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())
    pd.DataFrame([
        {"dataset": "Test", "slope": m_tr, "intercept": b_tr},
    ]).to_csv(DATA_DIR / "avg_lambda_test_set_25_linear_fit.csv", index=False)

    r_tr, n_tr = pearsonr_safe(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    ax.scatter(df_tr[ocn_tr], df_tr[y_sim_tr], s=6, alpha=0.35, c=COLOR_BLUE, edgecolors="none", label="Test")

    # Red limit lines
    for x0 in LAMBDA_INTERMEDIATE_BAND:
        ax.axvline(x0, **LIMIT_LINE_STYLE)

    # Draw straight lines across each dataset span
    for (m, b, xvals, col) in [
        (m_tr, b_tr, df_tr[ocn_tr].to_numpy(), COLOR_BLUE),
    ]:
        if np.isfinite(m) and np.isfinite(b) and xvals.size:
            xa, xb = np.nanmin(xvals), np.nanmax(xvals)
            ax.plot([xa, xb], [m*xa + b, m*xb + b], lw=1.0, color=col)

    ax.set_xlabel(LBL_X_AL)
    ax.set_ylabel(LBL_Y_SIM)
    ax.text(0.97, 0.03, f"Test-set: r={r_tr:.2f} (n={n_tr}))",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "avg_lambda_test_set_25_dependence")

def panel_B():
    df_tr = _read_csv(PAIRS25_CSV)


    y_sim_tr = find_col(df_tr, ["B22_corr1_log10", "y_sim", "log10_negB22_sim"], "simulated y (Test-Set)")
    ocn_tr = find_col(df_tr, ["Average_NCPR", "Avg_NCPR", "average_ncpr"], "Average NCPR (Test-Set)")
    pair_tr = get_pair_col(df_tr)


    # Save points
    pts_tr = df_tr[[pair_tr, ocn_tr, y_sim_tr]].copy()
    pts_tr.insert(0, "dataset", "Test")
    pts_tr.columns = ["dataset", "pair", "avg_ncpr", "y_sim"]

  
    pts = pd.concat([pts_tr,], ignore_index=True)
    pts.to_csv(DATA_DIR / "avg_ncpr_test_set_25_points.csv", index=False)

    # Also keep binned CSV (compat), but do not plot it
    med_tr = binned_median(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy(), n_bins=N_BINS)
    med_tr.insert(0, "dataset", "Test")
    # Linear fits (plotted)
    m_tr, b_tr = linear_fit_params(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())
    pd.DataFrame([
        {"dataset": "Test", "slope": m_tr, "intercept": b_tr},
    ]).to_csv(DATA_DIR / "avg_ncpr_test_set_25_linear_fit.csv", index=False)

    r_tr, n_tr = pearsonr_safe(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    ax.scatter(df_tr[ocn_tr], df_tr[y_sim_tr], s=6, alpha=0.35, c=COLOR_BLUE, edgecolors="none", label="Test")

    # Red limit lines
    for x0 in NCPR_NEUTRAL_BAND:
        ax.axvline(x0, **LIMIT_LINE_STYLE)

    # Draw straight lines across each dataset span
    for (m, b, xvals, col) in [
        (m_tr, b_tr, df_tr[ocn_tr].to_numpy(), COLOR_BLUE),
    ]:
        if np.isfinite(m) and np.isfinite(b) and xvals.size:
            xa, xb = np.nanmin(xvals), np.nanmax(xvals)
            ax.plot([xa, xb], [m*xa + b, m*xb + b], lw=1.0, color=col)

    ax.set_xlabel(LBL_X_ANCPR)
    ax.set_ylabel(LBL_Y_SIM)
    ax.text(0.97, 0.03, f"Test-set: r={r_tr:.2f} (n={n_tr}))",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "avg_ncpr_test_set_25_dependence")

def panel_C():
    df_tr = _read_csv(PAIRS25_CSV)


    y_sim_tr = find_col(df_tr, ["B22_corr1_log10", "y_sim", "log10_negB22_sim"], "simulated y (Test-Set)")
    ocn_tr = find_col(df_tr, ["Charge_Complementarity", "charge_complementarity", "charge_comp"], "Charge Complementarity (Test-Set)")
    pair_tr = get_pair_col(df_tr)

    # Save points
    pts_tr = df_tr[[pair_tr, ocn_tr, y_sim_tr]].copy()
    pts_tr.insert(0, "dataset", "Test")
    pts_tr.columns = ["dataset", "pair", "charge_complementarity", "y_sim"]

  
    pts = pd.concat([pts_tr,], ignore_index=True)
    pts.to_csv(DATA_DIR / "charge_complementarity_test_set_25_points.csv", index=False)

    # Also keep binned CSV (compat), but do not plot it
    med_tr = binned_median(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy(), n_bins=N_BINS)
    med_tr.insert(0, "dataset", "Test")
    # Linear fits (plotted)
    m_tr, b_tr = linear_fit_params(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())
    pd.DataFrame([
        {"dataset": "Test", "slope": m_tr, "intercept": b_tr},
    ]).to_csv(DATA_DIR / "charge_complementarity_test_set_25_linear_fit.csv", index=False)

    r_tr, n_tr = pearsonr_safe(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    ax.scatter(df_tr[ocn_tr], df_tr[y_sim_tr], s=6, alpha=0.35, c=COLOR_BLUE, edgecolors="none", label="Test")

    # Red limit line
    ax.axvline(CHARGE_COMPLEMENTARITY_MIN, **LIMIT_LINE_STYLE)

    # Draw straight lines across each dataset span
    for (m, b, xvals, col) in [
        (m_tr, b_tr, df_tr[ocn_tr].to_numpy(), COLOR_BLUE),
    ]:
        if np.isfinite(m) and np.isfinite(b) and xvals.size:
            xa, xb = np.nanmin(xvals), np.nanmax(xvals)
            ax.plot([xa, xb], [m*xa + b, m*xb + b], lw=1.0, color=col)

    ax.set_xlabel(LBL_X_CC)
    ax.set_ylabel(LBL_Y_SIM)
    ax.text(0.97, 0.03, f"Test-set: r={r_tr:.2f} (n={n_tr}))",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "charge_complementarity_test_set_25_dependence")

def panel_D():
    df_tr = _read_csv(INITIAL_SET_CSV)


    y_sim_tr = find_col(df_tr, ["B22_corr1_log10", "y_sim", "log10_negB22_sim"], "simulated y (Training-Initial)")
    ocn_tr = find_col(df_tr, ["Opposite_Charge_Number", "OCN", "ocn"], "Opposite Charge Number (Training-Initial)")
    pair_tr = get_pair_col(df_tr)

    # Save points
    pts_tr = df_tr[[pair_tr, ocn_tr, y_sim_tr]].copy()
    pts_tr.insert(0, "dataset", "Training")
    pts_tr.columns = ["dataset", "pair", "Opposite_Charge_Number", "y_sim"]

  
    pts = pd.concat([pts_tr,], ignore_index=True)
    pts.to_csv(DATA_DIR / "ocn_test_set_25_points.csv", index=False)

    # Also keep binned CSV (compat), but do not plot it
    med_tr = binned_median(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy(), n_bins=N_BINS)
    med_tr.insert(0, "dataset", "Training")
    # Linear fits (plotted)
    m_tr, b_tr = linear_fit_params(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())
    pd.DataFrame([
        {"dataset": "Training", "slope": m_tr, "intercept": b_tr},
    ]).to_csv(DATA_DIR / "ocn_test_set_25_linear_fit.csv", index=False)

    r_tr, n_tr = pearsonr_safe(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    ax.scatter(df_tr[ocn_tr], df_tr[y_sim_tr], s=6, alpha=0.35, c=COLOR_BLUE, edgecolors="none", label="Training")

    # Draw straight lines across each dataset span
    for (m, b, xvals, col) in [
        (m_tr, b_tr, df_tr[ocn_tr].to_numpy(), COLOR_BLUE),
    ]:
        if np.isfinite(m) and np.isfinite(b) and xvals.size:
            xa, xb = np.nanmin(xvals), np.nanmax(xvals)
            ax.plot([xa, xb], [m*xa + b, m*xb + b], lw=1.0, color=col)

    ax.set_xlabel(LBL_X_OCN)
    ax.set_ylabel(LBL_Y_SIM)
    ax.text(0.97, 0.03, f"Training-Initial: r={r_tr:.2f} (n={n_tr}))",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "ocn_test_set_25_dependence")



# =========================
# Main
# =========================
def main(argv: Optional[Iterable[str]] = None) -> None:
    global OUTPUT_DIR, DATA_DIR
    global TRAINING_CSV, HELDOUT_CSV, UNSIM_CSV, PAIRS25_CSV, INITIAL_SET_CSV
    global FEATURE_IMPORTANCE_CSV, RBP_UNIID_FILE

    p = argparse.ArgumentParser(description="Build supplementary Figure 4/5-style panels from the current workflow outputs.")
    p.add_argument("--outdir", type=Path, default=OUTPUT_DIR, help="Output directory. Default: sims_supplementary")
    p.add_argument("--pairs25-csv", type=str, default=PAIRS25_CSV, help="Main feature/B22 CSV for test-set style panels.")
    p.add_argument("--initial-set-csv", type=str, default=INITIAL_SET_CSV, help="Feature/B22 CSV for initial-set/OCN panel.")
    p.add_argument("--training-csv", type=str, default=TRAINING_CSV, help="Optional training CSV (kept for compatibility).")
    p.add_argument("--heldout-csv", type=str, default=HELDOUT_CSV, help="Optional held-out CSV (kept for compatibility).")
    p.add_argument("--unsim-csv", type=str, default=UNSIM_CSV, help="Optional unsimulated CSV (kept for compatibility).")
    p.add_argument("--feature-importance-csv", type=str, default=FEATURE_IMPORTANCE_CSV, help="Optional feature importance CSV.")
    p.add_argument("--rbp-uniid-file", type=str, default=RBP_UNIID_FILE, help="Optional RBP UniProt identifier list.")
    args = p.parse_args(list(argv) if argv is not None else None)

    OUTPUT_DIR = args.outdir
    DATA_DIR = OUTPUT_DIR / "data"
    PAIRS25_CSV = args.pairs25_csv
    INITIAL_SET_CSV = args.initial_set_csv
    TRAINING_CSV = args.training_csv
    HELDOUT_CSV = args.heldout_csv
    UNSIM_CSV = args.unsim_csv
    FEATURE_IMPORTANCE_CSV = args.feature_importance_csv
    RBP_UNIID_FILE = args.rbp_uniid_file

    ensure_dirs()

    print(f"[i] Output directory: {OUTPUT_DIR.resolve()}")
    print(f"[i] Data directory:   {DATA_DIR.resolve()}")
    print(f"[i] Pairs25 CSV:      {Path(PAIRS25_CSV).resolve()}")
    print(f"[i] Initial-set CSV:  {Path(INITIAL_SET_CSV).resolve()}")

    panel_A()
    print("Saved: avg_lambda_test_set_25_dependence.(pdf|png) and associated CSVs")

    panel_B()
    print("Saved: avg_ncpr_test_set_25_dependence.(pdf|png) and associated CSVs")

    panel_C()
    print("Saved: charge_complementarity_test_set_25_dependence.(pdf|png) and associated CSVs")

    panel_D()
    print("Saved: ocn_test_set_25_dependence.(pdf|png) and associated CSVs")

if __name__ == "__main__":
    main()
