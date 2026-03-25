#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Figure 4 (2×2) and save plotting tables to out/cache/.

This version is adapted to the current idr-interaction-analysis workflow.
It uses:
- results/analysis/pair_features_b22.csv as the master table
- RDF pickle files produced by save_rdfs.py / compute_b22.py
- energies_<pair>_<box>nm_sliced.pkl files produced by compute_pair_energies.py

Unlike older manuscript-side scripts, Panel C is built directly from the
energy PKLs and does not require amino_acids_energy_interactions.py.
The --aa_script argument is accepted for CLI compatibility but is not used.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


# ============================ Global Style ===================================
FONT_SCALE = 1.25

def _fs(pts: float) -> float:
    return float(pts) * float(FONT_SCALE)

AXIS_LABEL_FONTSIZE = _fs(6.5)
AXIS_TITLE_FONTSIZE = _fs(7.0)
TICK_LABEL_FONTSIZE = _fs(6.0)
LEGEND_FONTSIZE = _fs(6.0)
FIGSIZE_PANEL = (4.15, 2.65)

AXES_LINEWIDTH = 0.6
TICK_WIDTH = 0.5
LINE_WIDTH = 0.8
DPI_SAVE = 2400

_FALLBACK = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#7A7A7A",
    "grey_light": "#D3D3D3",
    "black": "#000000",
}
PALETTE: Dict[str, str] = {}


def C(name: str, fallback_key: str) -> str:
    try:
        if name in PALETTE:
            return PALETTE[name]
    except Exception:
        pass
    return _FALLBACK.get(fallback_key, "#000000")


def hide_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_axes(ax, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=AXIS_TITLE_FONTSIZE)
    for sp in ax.spines.values():
        sp.set_linewidth(AXES_LINEWIDTH)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE, width=TICK_WIDTH)
    hide_top_right(ax)


def style_legend(leg, size: float | None = None):
    if leg is None:
        return
    fs = LEGEND_FONTSIZE if size is None else size
    for txt in leg.get_texts():
        txt.set_fontsize(fs)
    if leg.get_title():
        leg.get_title().set_fontsize(fs)


def style_colorbar(cbar, label: str | None = None, ticksize: float | None = None):
    if label:
        cbar.set_label(label, fontsize=(LEGEND_FONTSIZE if ticksize is None else ticksize))
    cbar.ax.tick_params(labelsize=(TICK_LABEL_FONTSIZE if ticksize is None else ticksize), width=TICK_WIDTH)
    cbar.ax.yaxis.get_offset_text().set_size(TICK_LABEL_FONTSIZE if ticksize is None else ticksize)
    return cbar


def apply_mpl_style():
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["font.size"] = TICK_LABEL_FONTSIZE
    mpl.rcParams["axes.linewidth"] = AXES_LINEWIDTH
    mpl.rcParams["xtick.major.width"] = TICK_WIDTH
    mpl.rcParams["ytick.major.width"] = TICK_WIDTH
    mpl.rcParams["lines.linewidth"] = LINE_WIDTH
    mpl.rcParams["savefig.dpi"] = DPI_SAVE
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


# ================================ IO helpers =================================
def ensure_dirs(outdir: Path) -> tuple[Path, Path]:
    outdir = Path(outdir).expanduser().resolve()
    cache = outdir / "cache"
    outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return outdir, cache


def load_master_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path).expanduser().resolve())
    df.columns = [str(c).strip() for c in df.columns]

    # Case-insensitive normalization so the script works with both the old
    # manuscript naming and the new a/b feature-table naming.
    lower_to_actual = {c.lower(): c for c in df.columns}

    def rename_if_present(src: str, dst: str) -> None:
        actual = lower_to_actual.get(src.lower())
        if actual is not None and dst not in df.columns:
            df.rename(columns={actual: dst}, inplace=True)
            lower_to_actual[dst.lower()] = dst

    # New pipeline naming -> legacy names expected by this plotting script
    rename_if_present("a", "rgg")
    rename_if_present("b", "tad")
    rename_if_present("A_length", "RGG_length")
    rename_if_present("B_length", "TAD_length")
    rename_if_present("charge_A", "charge_RGG")
    rename_if_present("charge_B", "charge_TAD")
    rename_if_present("lambda_A", "lambda_RGG")
    rename_if_present("lambda_B", "lambda_TAD")
    rename_if_present("NCPR_A", "NCPR_RGG")
    rename_if_present("NCPR_B", "NCPR_TAD")
    rename_if_present("SCD_A", "SCD_RGG")
    rename_if_present("SCD_B", "SCD_TAD")
    rename_if_present("SHD_A", "SHD_RGG")
    rename_if_present("SHD_B", "SHD_TAD")

    # Additional tolerant aliases seen in older intermediate files
    rename_if_present("lambda_tad", "lambda_TAD")
    rename_if_present("lambda_tad ", "lambda_TAD")
    rename_if_present("lambda_b", "lambda_TAD")
    rename_if_present("lambda_rgg", "lambda_RGG")
    rename_if_present("lambda_a", "lambda_RGG")

    required = {"name", "Opposite_Charge_Number", "B22_corr1_log10", "Corrected_B22_Hummer", "lambda_TAD"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(
            f"Master CSV missing columns: {miss}. Available columns are: {list(df.columns)}"
        )
    return df


# =============================== Panel A =====================================
def panel_a(ax: plt.Axes, df: pd.DataFrame, cache: Path):
    vals = df["B22_corr1_log10"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    df[["name", "B22_corr1_log10"]].to_csv(cache / "Fig4a_B22_values.csv", index=False)

    bins = 70 if vals.size >= 300 else 40
    ax.hist(vals, bins=bins, color=C("grey_light", "grey_light"), edgecolor=C("grey", "grey"), linewidth=0.35)

    if vals.size:
        med = float(np.nanmedian(vals))
        p5, p95 = np.nanpercentile(vals, [5, 95])
        ax.axvline(med, color=C("black", "black"), linestyle="--", linewidth=0.7, label=f"Median = {round(med, 3)}")
        ax.axvline(p5, color=C("black", "black"), linestyle=":", linewidth=0.7, label="5–95%")
        ax.axvline(p95, color=C("black", "black"), linestyle=":", linewidth=0.7)

    style_axes(
        ax,
        xlabel=r"$\log_{10}\!\left(-B_{22}\right)$",
        ylabel="Number of Simulated Pairs",
        title=r"Distribution of $\log_{10}\!\left(-B_{22}\right)$",
    )
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    style_legend(ax.legend(frameon=False, loc="upper right"))


# =============================== Panel B =====================================
def load_rdf_curves_via_corrected_rdfs(pkl_path: Path, *, last: float, box_nm: float, delta: float) -> dict:
    from rdf_b22_functions import corrected_rdfs_b22

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and {"r_nm", "rdf_mean"}.issubset(payload.keys()):
        df = pd.DataFrame(
            {
                "run_1": [
                    np.asarray(payload["r_nm"], dtype=float),
                    np.asarray(payload["rdf_mean"], dtype=float),
                ]
            }
        )
    else:
        df = payload

    rdfs_dict, b22s_dict, kds_dict = corrected_rdfs_b22(df, last, box_size=box_nm, delta=delta)
    if not rdfs_dict:
        raise ValueError(f"rdfs_dict is empty for {pkl_path}")

    run_key = sorted(rdfs_dict.keys())[0]
    rdf_entry = rdfs_dict[run_key]
    return {
        "rs": np.asarray(rdf_entry["rs"], dtype=float),
        "orig": np.asarray(rdf_entry["rdfs"], dtype=float),
        "corr1": np.asarray(rdf_entry["rdfs_corr1"], dtype=float),
        "corr2": np.asarray(rdf_entry["rdfs_corr2"], dtype=float),
        "run_key": run_key,
    }


def panel_b(ax_parent: plt.Axes, rdf_strong: Path, rdf_weak: Path, cache: Path, *, box_nm: float, last: float, delta: float):
    ax_parent.axis("off")
    fig = ax_parent.figure
    bbox = ax_parent.get_position()

    pad_x = 0.03 * bbox.width
    mid_x = bbox.x0 + 0.5 * bbox.width
    left_w = (mid_x - bbox.x0) - pad_x / 2
    right_w = (bbox.x1 - mid_x) - pad_x / 2

    ax_left = fig.add_axes([bbox.x0, bbox.y0, left_w, bbox.height])
    ax_right = fig.add_axes([mid_x + pad_x / 2.0, bbox.y0, right_w, bbox.height])

    strong = load_rdf_curves_via_corrected_rdfs(rdf_strong, last=last, box_nm=box_nm, delta=delta)
    weak = load_rdf_curves_via_corrected_rdfs(rdf_weak, last=last, box_nm=box_nm, delta=delta)

    pd.DataFrame({"r_nm": strong["rs"], "g_orig": strong["orig"], "g_corr1": strong["corr1"], "g_corr2": strong["corr2"]}).to_csv(cache / "Fig4b_rdf_interacting.csv", index=False)
    pd.DataFrame({"r_nm": weak["rs"], "g_orig": weak["orig"], "g_corr1": weak["corr1"], "g_corr2": weak["corr2"]}).to_csv(cache / "Fig4b_rdf_noninteracting.csv", index=False)

    def _plot_one(ax, cur, color, *, xlabel: bool, ylabel: bool, title: str):
        (ln,) = ax.plot(cur["rs"], cur["corr1"], color=color, lw=1.1, label="RDF")
        ax.axhline(1.0, color=C("grey", "grey"), ls=":", lw=0.7)
        ax.set_xlim(0.0, box_nm / 2.0)
        ax.set_ylim(bottom=0.0)
        style_axes(ax, xlabel=(r"$r$ (nm)" if xlabel else None), ylabel=(r"$g(r)$" if ylabel else None), title=title)
        style_legend(ax.legend(handles=[ln], labels=["RDF"], frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.0), size=_fs(5.8))

    _plot_one(ax_left, strong, C("blue", "blue"), xlabel=True, ylabel=True, title="Interacting pair")
    _plot_one(ax_right, weak, C("grey", "grey"), xlabel=True, ylabel=False, title="Non-interacting pair")


# =============================== Panel C =====================================
AA_CLASS = {
    "R": "Basic", "K": "Basic", "H": "Basic",
    "D": "Acidic", "E": "Acidic",
    "F": "Aromatic", "Y": "Aromatic", "W": "Aromatic",
}
CLASS_ORDER = ["Basic", "Acidic", "Aromatic", "Other"]
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def aa1_from_resname(resname: str) -> str:
    key = str(resname).strip().upper()
    if len(key) == 1:
        return key
    return AA3_TO_1.get(key, "X")


def aa_class(resname: str) -> str:
    aa = aa1_from_resname(resname)
    return AA_CLASS.get(aa, "Other")


def panel_c(ax: plt.Axes, *, df: pd.DataFrame, energy_dir: Path, cache: Path, box_nm: int, clip: float = 2.0):
    energy_dir = Path(energy_dir).expanduser().resolve()
    if not energy_dir.exists():
        ax.text(0.5, 0.5, "Panel C unavailable\n(missing energy_dir)", ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE)
        style_axes(ax)
        return

    work = df[["name", "B22_corr1_log10"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        ax.text(0.5, 0.5, "Panel C unavailable\n(no B22 data)", ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE)
        style_axes(ax)
        return

    matrices = []
    missing = []
    for name in work["name"].tolist():
        p = energy_dir / f"energies_{name}_{box_nm}nm_sliced.pkl"
        if not p.exists():
            missing.append(name)
            continue
        with open(p, "rb") as f:
            payload = pickle.load(f)
        required = {"avg_total_per_pair", "res1", "res2"}
        if not (isinstance(payload, dict) and required.issubset(payload.keys())):
            missing.append(name)
            continue

        vals = np.asarray(payload["avg_total_per_pair"], dtype=float)
        res1_raw = list(payload["res1"])
        res2_raw = list(payload["res2"])

        if vals.ndim != 2:
            try:
                vals = vals.reshape(len(res1_raw), len(res2_raw))
            except Exception:
                missing.append(name)
                continue

        if vals.shape != (len(res1_raw), len(res2_raw)):
            missing.append(name)
            continue

        res1 = [aa_class(x) for x in res1_raw]
        res2 = [aa_class(x) for x in res2_raw]
        attractive = np.clip(-vals, 0.0, None)

        G = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=float)
        class_index = {c: i for i, c in enumerate(CLASS_ORDER)}
        for i, ci in enumerate(res1):
            for j, cj in enumerate(res2):
                G[class_index[ci], class_index[cj]] += attractive[i, j]
        matrices.append(G)

    pd.Series(missing, name="missing_energy_pkls").to_csv(cache / "Fig4c_missing_energy_pkls.csv", index=False)

    if not matrices:
        ax.text(0.5, 0.5, "Panel C unavailable\n(no usable energy PKLs)", ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE)
        style_axes(ax)
        return

    G_obs = np.mean(np.stack(matrices, axis=0), axis=0)
    row = G_obs.sum(axis=1, keepdims=True)
    col = G_obs.sum(axis=0, keepdims=True)
    tot = G_obs.sum()
    G_exp = np.zeros_like(G_obs) if tot <= 0 else (row @ col) / float(tot)
    eps = 1e-9
    G_log2 = np.log2((G_obs + eps) / (G_exp + eps))
    G_plot = np.clip(G_log2, -clip, clip)

    pd.DataFrame(G_obs, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(cache / "Fig4c_observed_classes.csv")
    pd.DataFrame(G_exp, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(cache / "Fig4c_expected_classes.csv")
    pd.DataFrame(G_log2, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(cache / "Fig4c_log2_enrichment_classes.csv")

    im = ax.imshow(G_plot, origin="lower", cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="equal")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=45, ha="right", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_yticklabels(CLASS_ORDER, fontsize=TICK_LABEL_FONTSIZE)
    style_axes(ax, xlabel="TAD residue class", ylabel="RGG residue class", title=rf"Residue-class enrichment (log$_2$ obs/exp)")

    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            v = G_log2[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=_fs(5.0), color=C("grey", "grey"))

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    style_colorbar(plt.colorbar(im, cax=cax), label=r"$\log_2(\mathrm{obs/exp})$")


# =============================== Panel D =====================================
def panel_d(ax: plt.Axes, df: pd.DataFrame, cache: Path):
    table = df[["name", "Opposite_Charge_Number", "lambda_TAD", "B22_corr1_log10"]].replace([np.inf, -np.inf], np.nan).dropna()
    table.to_csv(cache / "Fig4d_scatter_table.csv", index=False)

    x = table["Opposite_Charge_Number"].to_numpy(dtype=float)
    y = table["B22_corr1_log10"].to_numpy(dtype=float)
    c = table["lambda_TAD"].to_numpy(dtype=float)

    sc = ax.scatter(x, y, c=c, s=10, alpha=0.6, linewidths=0.0, cmap="viridis")
    style_axes(ax, xlabel="Opposite-charge number (OCN)", ylabel=r"$\log_{10}\!\left(-B_{22}\right)$", title=r"$\log_{10}\!\left(-B_{22}\right)$" + " Vs. OCN: Simulated pairs")

    if x.size >= 3:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ax.plot(xx, intercept + slope * xx, color=C("black", "black"), lw=0.9, alpha=0.85)
        ax.text(
            0.02,
            0.98,
            f"Pearson r = {r:.2f}\n(n = {x.size})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_fs(5.4),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C("grey", "grey"), lw=0.5, alpha=0.9),
        )

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    style_colorbar(plt.colorbar(sc, cax=cax), label=r"$\lambda_{\mathrm{TAD}}$")


# =============================== Compose =====================================
def build_figure(df: pd.DataFrame, *, rdf_strong: Path, rdf_weak: Path, energy_dir: Path, outdir: Path, cache: Path, box_nm: int, rdf_last: float, rdf_delta: float):
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    axA = fig.add_subplot(gs[0, 0])
    axB_parent = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    panel_a(axA, df, cache)
    panel_b(axB_parent, rdf_strong, rdf_weak, cache, box_nm=float(box_nm), last=rdf_last, delta=rdf_delta)
    panel_c(axC, df=df, energy_dir=energy_dir, cache=cache, box_nm=box_nm, clip=2.0)
    panel_d(axD, df, cache)
    fig.subplots_adjust(bottom=0.15)

    fig.savefig(outdir / "Figure4_full.pdf")
    fig.savefig(outdir / "Figure4_full.png", dpi=DPI_SAVE)
    plt.close(fig)


def build_panel_b_figure(rdf_strong: Path, rdf_weak: Path, cache: Path, out_panels: Path, *, box_nm: int, rdf_last: float, rdf_delta: float):
    fig = plt.figure(figsize=FIGSIZE_PANEL)
    ax_parent = fig.add_axes([0.3, 0.3, 0.88, 0.88])
    panel_b(ax_parent, rdf_strong, rdf_weak, cache, box_nm=float(box_nm), last=rdf_last, delta=rdf_delta)
    fig.savefig(out_panels / "Figure4_panel_b.pdf")
    fig.savefig(out_panels / "Figure4_panel_b.png", dpi=DPI_SAVE)
    plt.close(fig)


# =============================== CLI =========================================
def main(argv: Optional[List[str]] = None):
    apply_mpl_style()

    p = argparse.ArgumentParser(description="Build Figure 4 (2×2) and save plotting tables to out/cache/")
    p.add_argument("--table", type=Path, default=Path("results/analysis/pair_features_b22.csv"), help="Master CSV with features + B22 columns.")
    p.add_argument("--outdir", type=Path, default=Path("results/figures/fig4"), help="Output directory.")
    p.add_argument("--rdf_strong", type=Path, required=True, help="Pickle file for strong RDF example.")
    p.add_argument("--rdf_weak", type=Path, required=True, help="Pickle file for weak RDF example.")
    p.add_argument("--energy_dir", type=Path, default=Path("test_energies"), help="Directory containing energies_{name}_{box}nm_sliced.pkl files.")
    p.add_argument("--aa_script", type=Path, default=Path("scripts/amino_acids_energy_interactions.py"), help="Accepted for CLI compatibility but not used in this script.")
    p.add_argument("--box_nm", type=int, default=20, help="Simulation box (nm) used in energy pickle filenames.")
    p.add_argument("--rdf_last", type=float, default=7.5, help="Upper integration limit (nm) for vertical lines.")
    p.add_argument("--rdf_delta", type=float, default=2.0, help="Delta shift for second vertical line.")
    p.add_argument("--export_metrics", action="store_true", help="Accepted for CLI compatibility; ignored in this script.")
    p.add_argument("--metrics_csv", type=Path, default=None, help="Unused placeholder for CLI compatibility.")
    args = p.parse_args(argv)

    outdir, cache = ensure_dirs(args.outdir)
    df = load_master_table(args.table)

    rdf_strong = Path(args.rdf_strong).expanduser().resolve()
    rdf_weak = Path(args.rdf_weak).expanduser().resolve()
    if not rdf_strong.exists():
        raise FileNotFoundError(f"Missing rdf_strong: {rdf_strong}")
    if not rdf_weak.exists():
        raise FileNotFoundError(f"Missing rdf_weak: {rdf_weak}")

    energy_dir = Path(args.energy_dir).expanduser().resolve()

    build_figure(
        df,
        rdf_strong=rdf_strong,
        rdf_weak=rdf_weak,
        energy_dir=energy_dir,
        outdir=outdir,
        cache=cache,
        box_nm=args.box_nm,
        rdf_last=args.rdf_last,
        rdf_delta=args.rdf_delta,
    )
    build_panel_b_figure(rdf_strong, rdf_weak, cache, outdir, box_nm=args.box_nm, rdf_last=args.rdf_last, rdf_delta=args.rdf_delta)
    print(f"Done. Wrote: {outdir / 'Figure4_full.pdf'} and {outdir / 'Figure4_full.png'}")


if __name__ == "__main__":
    main()
