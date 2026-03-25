# ===== Amino-acid specific interactions across multiple CALVADOS simulations =====
# Uses your calc_per_pair_average_energy(...) exactly as-is.
# Output: per-simulation dicts (RGG/TAD AA contributions) + a tidy DataFrame
# for easy comparison/plotting across all simulations.
from per_pair_energies import calc_per_pair_average_energy, plot_energy_matrix, calc_avg_energies
from MDAnalysis.coordinates import DCD
import MDAnalysis as mda
# ===== RGG–TAD amino-acid comparison & metrics from saved PKLs =====
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# ----------------------------- Helpers ---------------------------------------

AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

AA_GROUPS = {
    "basic": set("RKH"),
    "acidic": set("DE"),
    "aromatic": set("FYW"),
    "polar": set("STNQCH"),
    "hydrophobic": set("AVILM"),
    "special": set("PG")
}

def _to_one_letter(names):
    out = []
    for x in list(names):
        x = str(x).upper()
        out.append(AA3_TO_1.get(x, x if len(x) == 1 else "X"))  # unknown -> "X"
    return np.array(out, dtype=object)

def _score(e, mode):
    """
    mode:
      - 'attractive': only E<0 counted, returned positive (stronger attraction = larger)
      - 'magnitude' : |E|
      - 'net'       : signed E
    """
    if mode == "attractive":
        return -e if e < 0.0 else 0.0
    if mode == "magnitude":
        return abs(e)
    return e

def _guess_is_rgg(resnames_1letter, thresh=0.6):
    """Heuristic: RGG segments are R/G-heavy."""
    if len(resnames_1letter) == 0:
        return False
    frac_rg = sum(r in ("R", "G") for r in resnames_1letter) / len(resnames_1letter)
    return frac_rg >= thresh

def _ensure_1d_pair_vector(x, n1, n2):
    """
    Accept (pairs,), (frames,pairs), (pairs,frames), or (n1,n2) and return a 1D vector of length n1*n2.
    For 2D frame-wise inputs, average across frames.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape == (n1, n2):
        return x.reshape(n1 * n2)
    if x.ndim == 2:
        if x.shape[1] == n1 * n2:   # frames × pairs
            return np.nanmean(x, axis=0)
        if x.shape[0] == n1 * n2:   # pairs × frames
            return np.nanmean(x, axis=1)
    return x.reshape(-1)

def parse_pair_name(name):
    """
    Parse pair names. Your convention is '{rgg__tad}' (double underscore).
    Fallback to single '_' split if '__' isn't present.
    """
    if "__" in name:
        return name.split("__", 1)
    # fallback for old files like 'rgg_tad'
    return name.split("_", 1)

# ---------------- Per-pickle collapse to AA dicts -----------------------------

def aa_dicts_from_pickle_arrays(ah_energies_pp, yu_energies_pp, res1, res2,
                                mode="attractive", normalize=True):
    """
    Collapse per-pair energies (AH+YU) to per-AA contributions for side A and side B.
    Returns: contrib_A, contrib_B, pair_mat, counts_A, counts_B
    """
    res1 = _to_one_letter(res1)
    res2 = _to_one_letter(res2)
    n1, n2 = len(res1), len(res2)

    ah = _ensure_1d_pair_vector(ah_energies_pp, n1, n2)
    yu = _ensure_1d_pair_vector(yu_energies_pp, n1, n2)
    E  = np.nan_to_num(ah) + np.nan_to_num(yu)

    contrib_A = defaultdict(float)
    contrib_B = defaultdict(float)
    pair_mat  = defaultdict(float)

    # row-major: i over res1, j over res2
    for k, e in enumerate(E):
        i = k // n2
        j = k %  n2
        a = res1[i]; b = res2[j]
        v = _score(e, mode)
        contrib_A[a] += v
        contrib_B[b] += v
        pair_mat[(a, b)] += v

    counts_A = Counter(res1)
    counts_B = Counter(res2)

    if normalize:
        for aa in list(contrib_A.keys()):
            contrib_A[aa] /= max(counts_A[aa], 1)
        for bb in list(contrib_B.keys()):
            contrib_B[bb] /= max(counts_B[bb], 1)

    contrib_A = dict(sorted(contrib_A.items(), key=lambda x: x[1], reverse=True))
    contrib_B = dict(sorted(contrib_B.items(), key=lambda x: x[1], reverse=True))
    return contrib_A, contrib_B, pair_mat, counts_A, counts_B

# ---------------- Name-aware batch analysis ----------------------------------

def analyze_energy_pkls_with_names(
    names, out_folder, box,
    mode="attractive", normalize=True,
    assume_name_order=False,   # True if segid A is always the first token (RGG)
    rgg_guess_thresh=0.6
):
    """
    Loads '{out_folder}/energies_{name}_{box}nm_sliced.pkl' for each name='{rgg}_{tad}',
    collapses to per-AA contributions, and returns:
      - results[name]: dict with 'RGG','TAD' (AA->value), plus metadata
      - df: tidy DataFrame with columns:
            [label, rgg_name, tad_name, side, aa, contribution, mode, normalized]
    """
    results = {}
    rows = []

    for name in names:
        rgg_label, tad_label = parse_pair_name(name)
        pkl_path = os.path.join(out_folder, f"energies_{name}_{box}nm_sliced.pkl")
        if not os.path.exists(pkl_path):
            print(f"[warn] missing: {pkl_path} — skipping")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        ah = data["ah_energies_pp"]
        yu = data["yu_energies_pp"]
        res1 = data["res1"]   # segid A residue names
        res2 = data["res2"]   # segid B residue names

        A_dict, B_dict, pair_mat, counts_A, counts_B = aa_dicts_from_pickle_arrays(
            ah, yu, res1, res2, mode=mode, normalize=normalize
        )

        # Decide which side is RGG vs TAD
        if assume_name_order:
            RGG, TAD = A_dict, B_dict
        else:
            resA = _to_one_letter(res1); resB = _to_one_letter(res2)
            isA = _guess_is_rgg(resA, rgg_guess_thresh)
            isB = _guess_is_rgg(resB, rgg_guess_thresh)
            if isA and not isB:
                RGG, TAD = A_dict, B_dict
            elif isB and not isA:
                RGG, TAD = B_dict, A_dict
            else:
                # ambiguous -> fall back to filename order to be deterministic
                RGG, TAD = A_dict, B_dict

        # Store per-sim result
        results[name] = {
            "rgg_name": rgg_label,
            "tad_name": tad_label,
            "RGG": RGG,
            "TAD": TAD,
            "side_A": A_dict, "side_B": B_dict,
            "aa_pair_matrix": pair_mat,
            "counts": {"A": counts_A, "B": counts_B},
            "mode": mode, "normalized": normalize
        }

        # Tidy rows (with names carried through for grouping)
        for side, dd in (("RGG", RGG), ("TAD", TAD)):
            for aa, val in dd.items():
                rows.append({
                    "label": name,
                    "rgg_name": rgg_label,
                    "tad_name": tad_label,
                    "side": side,
                    "aa": aa,
                    "contribution": float(val),
                    "mode": mode,
                    "normalized": normalize
                })

    df = pd.DataFrame(rows)
    return results, df

# ---------------- Build AA×AA heatmap across sims -----------------------------

def load_and_orient_pairmap(name, out_folder, box,
                            mode="attractive",
                            assume_name_order=False,
                            rgg_guess_thresh=0.6):
    """
    Returns for a single simulation:
      pair_sum[(RGG_AA, TAD_AA)] -> sum of contributions (this sim)
      pair_cnt[(RGG_AA, TAD_AA)] -> number of residue-residue pairs contributing
      side_dicts: {"RGG": {AA:val}, "TAD": {AA:val}} (composition-normalized)
    """
    pkl_path = os.path.join(out_folder, f"energies_{name}_{box}nm_sliced.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    ah = data["ah_energies_pp"]
    yu = data["yu_energies_pp"]
    res1 = data["res1"]  # segid A
    res2 = data["res2"]  # segid B

    r1 = _to_one_letter(res1)
    r2 = _to_one_letter(res2)
    n1, n2 = len(r1), len(r2)

    E = _ensure_1d_pair_vector(ah, n1, n2) + _ensure_1d_pair_vector(yu, n1, n2)
    E = np.nan_to_num(E)

    # Decide orientation (which side is RGG)
    if assume_name_order:
        rgg_is_A = True
    else:
        isA = _guess_is_rgg(r1, rgg_guess_thresh)
        isB = _guess_is_rgg(r2, rgg_guess_thresh)
        if isA and not isB:
            rgg_is_A = True
        elif isB and not isA:
            rgg_is_A = False
        else:
            rgg_is_A = True  # deterministic fallback

    pair_sum = defaultdict(float)
    pair_cnt = defaultdict(int)
    RGG_contrib = defaultdict(float)
    TAD_contrib = defaultdict(float)
    cnt_RGG = Counter(r1 if rgg_is_A else r2)
    cnt_TAD = Counter(r2 if rgg_is_A else r1)

    # Go through residue pairs
    for k, e in enumerate(E):
        i = k // n2
        j = k %  n2
        v = _score(e, mode)
        if rgg_is_A:
            a = r1[i]; b = r2[j]
        else:
            a = r2[j]; b = r1[i]
        if a == "X" or b == "X":  # skip unknowns in the AA×AA map
            continue
        pair_sum[(a, b)] += v
        pair_cnt[(a, b)] += 1
        RGG_contrib[a] += v
        TAD_contrib[b] += v

    # composition-normalize per side
    for aa in list(RGG_contrib.keys()):
        RGG_contrib[aa] /= max(cnt_RGG[aa], 1)
    for aa in list(TAD_contrib.keys()):
        TAD_contrib[aa] /= max(cnt_TAD[aa], 1)

    RGG_contrib = dict(sorted(RGG_contrib.items(), key=lambda x: x[1], reverse=True))
    TAD_contrib = dict(sorted(TAD_contrib.items(), key=lambda x: x[1], reverse=True))
    side_dicts = {"RGG": RGG_contrib, "TAD": TAD_contrib}
    return pair_sum, pair_cnt, side_dicts

def aggregate_RGG_TAD_AA_pair_heatmap(names, out_folder, box,
                                      mode="attractive",
                                      assume_name_order=False,
                                      rgg_guess_thresh=0.6,
                                      average_mode="mean_of_means"):
    """
    Build a matrix M[rggaa, tadaa] of average interaction strengths across simulations.

    average_mode:
      - "mean_of_means": per-sim mean per AA-pair (sum/count) then averaged across sims
      - "pooled": sums and counts pooled across all sims, then one global mean
    """
    AAs = list("ACDEFGHIKLMNPQRSTVWY")
    idxA = {aa: i for i, aa in enumerate(AAs)}
    idxB = idxA.copy()
    sum_M = np.zeros((20, 20), dtype=float)
    cnt_M = np.zeros((20, 20), dtype=float)

    per_sim_Ms = []

    for name in names:
        ps, pc, _ = load_and_orient_pairmap(
            name, out_folder, box, mode=mode,
            assume_name_order=assume_name_order,
            rgg_guess_thresh=rgg_guess_thresh
        )
        # This sim's pair_sum/count → temporary matrices
        Msum = np.zeros((20, 20), dtype=float)
        Mcnt = np.zeros((20, 20), dtype=float)
        for (a, b), s in ps.items():
            if a not in idxA or b not in idxB:
                continue
            i, j = idxA[a], idxB[b]
            Msum[i, j] += s
            Mcnt[i, j] += pc[(a, b)]
        if average_mode == "mean_of_means":
            with np.errstate(invalid='ignore', divide='ignore'):
                Mmean = np.where(Mcnt > 0, Msum / Mcnt, 0.0)
            per_sim_Ms.append(Mmean)
        else:
            sum_M += Msum
            cnt_M += Mcnt

    if average_mode == "mean_of_means":
        if len(per_sim_Ms) == 0:
            M = np.zeros((20, 20))
        else:
            M = np.mean(np.stack(per_sim_Ms, axis=0), axis=0)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            M = np.where(cnt_M > 0, sum_M / cnt_M, 0.0)

    return M, AAs, AAs  # rows: RGG AAs, cols: TAD AAs

# --- replace your heatmap plotter with this ---
def plot_RGG_TAD_AA_heatmap(M, r_labels, t_labels, title=None, outfile=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, origin="lower", aspect="equal")

    # tick labels
    ax.set_xticks(range(len(t_labels))); ax.set_xticklabels(t_labels, rotation=90)
    ax.set_yticks(range(len(r_labels))); ax.set_yticklabels(r_labels)

    # CLEAR AXIS LABELS: rows = RGG, cols = TAD
    ax.set_ylabel("RGG amino acids (rows)")
    ax.set_xlabel("TAD amino acids (columns)")

    # colorbar + title
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Average interaction strength")
    if title:
        ax.set_title(title + "  [rows=RGG, cols=TAD]")
    else:
        ax.set_title("RGG × TAD AA average interaction  [rows=RGG, cols=TAD]")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------- Compressed metrics & similarity -----------------------------

def _gini(values):
    x = np.sort(np.array(values, dtype=float))
    s = x.sum()
    if s <= 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    g = 1 + 1/n - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * x) / (n * s)
    return float(max(0.0, min(1.0, g)))

def _entropy(values):
    x = np.array(values, dtype=float)
    s = x.sum()
    if s <= 0:
        return 0.0
    p = x / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def _group_share(contrib_dict, group_set):
    total = sum(contrib_dict.values())
    if total <= 0:
        return 0.0
    num = sum(v for aa, v in contrib_dict.items() if aa in group_set)
    return float(num / total)

def compute_metrics_for_one_sim(side_dicts, topk=3):
    """
    side_dicts: {"RGG": {AA:val}, "TAD": {AA:val}}
    Returns a flat dict of metrics for both sides.
    """
    out = {}
    for side in ("RGG", "TAD"):
        d = side_dicts[side]
        vals = list(d.values())
        total = float(sum(vals))
        top_cov = float(sum(sorted(vals, reverse=True)[:topk]) / total) if total > 0 else 0.0
        out[f"{side}_total"] = total
        out[f"{side}_top{topk}_coverage"] = top_cov
        out[f"{side}_entropy"] = _entropy(vals)
        out[f"{side}_gini"] = _gini(vals)
        out[f"{side}_aromatic_share"] = _group_share(d, AA_GROUPS["aromatic"])
        out[f"{side}_acidic_share"] = _group_share(d, AA_GROUPS["acidic"])
        out[f"{side}_basic_share"] = _group_share(d, AA_GROUPS["basic"])
        # charge balance index: (basic - acidic) / (basic + acidic)
        b = _group_share(d, AA_GROUPS["basic"]) * total
        a = _group_share(d, AA_GROUPS["acidic"]) * total
        denom = (b + a) if (b + a) > 0 else 1.0
        out[f"{side}_charge_balance"] = float((b - a) / denom)
    return out

def compute_metrics_table(names, out_folder, box,
                          mode="attractive",
                          assume_name_order=False,
                          rgg_guess_thresh=0.6,
                          topk=3):
    """
    Returns a DataFrame with one row per simulation and metrics columns.
    """
    rows = []
    for name in names:
        _, _, side_dicts = load_and_orient_pairmap(
            name, out_folder, box, mode=mode,
            assume_name_order=assume_name_order,
            rgg_guess_thresh=rgg_guess_thresh
        )
        m = compute_metrics_for_one_sim(side_dicts, topk=topk)
        m["label"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("label").sort_index()

def cosine_similarity_between_sims(names, out_folder, box,
                                   side="RGG",
                                   mode="attractive",
                                   assume_name_order=False,
                                   rgg_guess_thresh=0.6):
    """
    Returns a DataFrame of cosine similarities between AA contribution vectors
    for the requested side ('RGG' or 'TAD') across simulations.
    """
    AAs = list("ACDEFGHIKLMNPQRSTVWY")
    vecs = {}
    for name in names:
        _, _, side_dicts = load_and_orient_pairmap(
            name, out_folder, box, mode=mode,
            assume_name_order=assume_name_order,
            rgg_guess_thresh=rgg_guess_thresh
        )
        d = side_dicts[side]
        v = np.array([d.get(aa, 0.0) for aa in AAs], dtype=float)
        norm = np.linalg.norm(v)
        vecs[name] = v / norm if norm > 0 else v
    keys = list(vecs.keys())
    M = np.zeros((len(keys), len(keys)))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = float(np.dot(vecs[ki], vecs[kj]))
    return pd.DataFrame(M, index=keys, columns=keys)


AAs20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_GROUPS = {
    "basic": ["R","K","H"],
    "acidic": ["D","E"],
    "aromatic": ["F","Y","W"],
    "polar": ["S","T","N","Q","C","H"],
    "hydrophobic": ["A","V","I","L","M"],
    "special": ["P","G"],
}

def per_sim_AA_pair_matrices(names, out_folder, box,
                             mode="attractive",
                             assume_name_order=False,
                             rgg_guess_thresh=0.6):
    """
    Build per-simulation AA×AA mean matrices (rows=RGG AA, cols=TAD AA).
    Returns: Ms (n_sims, 20, 20), labels, row_labels, col_labels
    """
    Ms = []
    labels = []
    for name in names:
        ps, pc, _ = load_and_orient_pairmap(
            name, out_folder, box, mode=mode,
            assume_name_order=assume_name_order,
            rgg_guess_thresh=rgg_guess_thresh
        )
        # convert pair_sums / counts to mean matrix for this sim
        Msum = np.zeros((20,20), dtype=float)
        Mcnt = np.zeros((20,20), dtype=float)
        idx = {aa:i for i,aa in enumerate(AAs20)}
        for (a,b), s in ps.items():
            if a in idx and b in idx:
                i, j = idx[a], idx[b]
                Msum[i,j] += s
                Mcnt[i,j] += pc[(a,b)]
        with np.errstate(invalid='ignore', divide='ignore'):
            Mmean = np.where(Mcnt>0, Msum/Mcnt, 0.0)
        Ms.append(Mmean)
        labels.append(name)
    if len(Ms)==0:
        Ms = np.zeros((0,20,20))
    else:
        Ms = np.stack(Ms, axis=0)
    return Ms, labels, AAs20, AAs20

# ---------- Metrics ----------

def independence_expected(M):
    """Expected matrix under row/col independence: E[i,j] = (row_i * col_j)/total."""
    row = M.sum(axis=1, keepdims=True)
    col = M.sum(axis=0, keepdims=True)
    total = float(row.sum())
    if total <= 0: 
        return np.zeros_like(M)
    return (row @ col) / total

def log2_enrichment(M, eps=1e-9):
    """log2( observed / expected ) using independence model (composition-aware)."""
    E = independence_expected(M)
    return np.log2((M + eps) / (E + eps)), E

def zscore_map(Ms):
    """
    Given per-sim matrices Ms (n,20,20), return Z = mean/sd (per cell).
    Good to see consistency across simulations.
    """
    if Ms.shape[0] < 2:
        return np.zeros_like(Ms[0])
    mu = Ms.mean(axis=0)
    sd = Ms.std(axis=0, ddof=1)
    Z = np.where(sd>0, mu/sd, 0.0)
    return Z

def condition_difference(names_A, names_B, out_folder, box,
                         mode="attractive", assume_name_order=False, rgg_guess_thresh=0.6):
    """Mean(A) - Mean(B) matrices and the two means."""
    MsA, _, _, _ = per_sim_AA_pair_matrices(names_A, out_folder, box, mode, assume_name_order, rgg_guess_thresh)
    MsB, _, _, _ = per_sim_AA_pair_matrices(names_B, out_folder, box, mode, assume_name_order, rgg_guess_thresh)
    MA = MsA.mean(axis=0) if MsA.size else np.zeros((20,20))
    MB = MsB.mean(axis=0) if MsB.size else np.zeros((20,20))
    return MA - MB, MA, MB

def collapse_groups(M, groups=AA_GROUPS):
    """
    Collapse 20×20 into G×G by averaging cells inside each group block.
    Returns: Mg, gnames
    """
    gnames = list(groups.keys())
    gmap = {aa:g for g,aas in groups.items() for aa in aas}
    idx = {aa:i for i,aa in enumerate(AAs20)}
    G = len(gnames)
    Mg = np.zeros((G,G))
    Cg = np.zeros((G,G))
    for i, ga in enumerate(gnames):
        for j, gb in enumerate(gnames):
            rows = [idx[aa] for aa in groups[ga] if aa in idx]
            cols = [idx[aa] for aa in groups[gb] if aa in idx]
            if not rows or not cols:
                continue
            block = M[np.ix_(rows, cols)]
            Mg[i,j] = block.mean()
            Cg[i,j] = block.size
    return Mg, gnames

# ---------- Plotting ----------

def _cluster_orders(M, metric='euclidean', method='average'):
    """Get row/col order via hierarchical clustering; fallback = sort by sums."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        Zr = linkage(M, method=method, metric=metric)
        Zc = linkage(M.T, method=method, metric=metric)
        r_ord = leaves_list(Zr)
        c_ord = leaves_list(Zc)
    except Exception:
        r_ord = np.argsort(-M.sum(axis=1))
        c_ord = np.argsort(-M.sum(axis=0))
    return r_ord, c_ord

def _plot_matrix(M, row_labels, col_labels, title, cbar_label, diverging=False, outfile=None):
    fig, ax = plt.subplots(figsize=(7,6))
    if diverging:
        vmax = np.nanmax(np.abs(M))
        vmin = -vmax
        im = ax.imshow(M, origin="lower", aspect="equal", vmin=vmin, vmax=vmax, cmap="coolwarm")
    else:
        im = ax.imshow(M, origin="lower", aspect="equal")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=90)
    ax.set_ylabel("RGG amino acids (rows)")
    ax.set_xlabel("TAD amino acids (columns)")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(cbar_label)
    ax.set_title(title)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight"); plt.close()
    else:
        plt.show()

def plot_clustered_heatmap(M, row_labels=AAs20, col_labels=AAs20,
                           title="AA×AA (clustered)", cbar_label="value",
                           diverging=False, outfile=None):
    r_ord, c_ord = _cluster_orders(M)
    _plot_matrix(M[np.ix_(r_ord, c_ord)],
                 [row_labels[i] for i in r_ord],
                 [col_labels[j] for j in c_ord],
                 title, cbar_label, diverging=diverging, outfile=outfile)

def top_pairs(M, row_labels=AAs20, col_labels=AAs20, k=20):
    """Return top-k AA pairs as a DataFrame."""
    flat_idx = np.argsort(M, axis=None)[::-1]
    rows = []
    for idx in flat_idx[:k]:
        i, j = np.unravel_index(idx, M.shape)
        rows.append({"RGG_AA": row_labels[i], "TAD_AA": col_labels[j], "value": float(M[i,j])})
    return pd.DataFrame(rows)





out_folder = "/Users/yuktikhanna/calvados/test_energies"
rggs = ["Q14152_5_3", "O00144_1_1", "Q9Y3Y2_3_1", "P19338_5_1", "Q02388_5_13"]
tads = ["P98177_463_504", "Q13285_373_392", "P49910_193_208", "Q9UJU2_14_65", "P35716_353_367"]
names = []
'''for i in rggs:
    for j in tads:
        names.append(f"{i}_{j}")'''

rggs_2 = ["nter", "thoc4_cter", "idr", "gar_plus100", "gar_plus50", "gar_plus20", "gar"]
#for i in rggs_2:
#    names.append(f"{i}_tad12")

def file_to_list(input_file):
    f1=open(input_file,"r")
    list1=[]
    for line in f1:
        stripped=line.strip()
        list1.append(stripped)
    f1.close()
    return list1

names = file_to_list("/Users/yuktikhanna/calvados/lists/sim_pairs_for_training_220425.txt")
box = 20
avg_energies = {}


# ---------------- Example usage (commented) -----------------------------------
#
out_folder = "/Users/yuktikhanna/calvados/test_energies"
box = 20  # whatever you used in filenames
#
# # 1) Per-simulation AA dicts + tidy table
results, df = analyze_energy_pkls_with_names(
     names, out_folder, box,
     mode="attractive",      # or "magnitude" / "net"
     normalize=True,
     assume_name_order=False # True if segid A is guaranteed the RGG side in files
)
df.to_csv(out_folder+"/aa_interactions_all_sims_named.csv", index=False)
#
# # 2) Heatmap across simulations (RGG-AA × TAD-AA)
M, r_labels, t_labels = aggregate_RGG_TAD_AA_pair_heatmap(
     names, out_folder, box,
     mode="attractive",
     assume_name_order=False,
     rgg_guess_thresh=0.6,
     average_mode="mean_of_means"  # or "pooled"
)
plot_RGG_TAD_AA_heatmap(
     M, r_labels, t_labels,
     title="RGG AA vs TAD AA — average interaction strength",
     outfile=out_folder+"/RGGxTAD_AA_avg_heatmap.png"
)
#
# # 3) Compressed metrics per simulation
metrics_df = compute_metrics_table(
     names, out_folder, box,
     mode="attractive",
     assume_name_order=False,
     rgg_guess_thresh=0.6,
     topk=3
)
metrics_df.to_csv(out_folder+"/aa_interaction_metrics_per_sim.csv")
#
# # 4) Similarity between simulations (fingerprints on RGG side)
simsim = cosine_similarity_between_sims(
     names, out_folder, box,
     side="RGG",
     mode="attractive",
     assume_name_order=False
)
simsim.to_csv(out_folder+"/RGG_side_cosine_similarity.csv")
#
# # Access a specific sim
# pair = names[0]
# print(results[pair]["rgg_name"], results[pair]["tad_name"])
# print("RGG dict:", results[pair]["RGG"])
# print("TAD dict:", results[pair]["TAD"])


# 1) Build per-sim matrices from your PKLs
Ms, labels, rlabs, clabs = per_sim_AA_pair_matrices(
    names, out_folder, box,
    mode="attractive",
    assume_name_order=False
)

# 2) Mean matrix across sims (the “plain” heatmap you already have)
Mmean = Ms.mean(axis=0) if Ms.size else np.zeros((20,20))
plot_clustered_heatmap(
    Mmean, rlabs, clabs,
    title="AA×AA mean interaction (clustered)  [rows=RGG, cols=TAD]",
    cbar_label="Avg interaction (kJ/mol, mode=attractive)",
    diverging=False,
    outfile=out_folder+"/AAxAA_mean_clustered.png"
)

# 3) Composition-aware **log2-enrichment** (observed vs expected independence)
Mlog2, Mexp = log2_enrichment(Mmean)
plot_clustered_heatmap(
    Mlog2, rlabs, clabs,
    title="AA×AA log2-enrichment over independence (clustered)",
    cbar_label="log2(observed / expected)",
    diverging=True,
    outfile=out_folder+"/AAxAA_log2_enrichment.png"
)

# 4) **Z-score map** across simulations (consistency/signal-to-noise)
Z = zscore_map(Ms)
plot_clustered_heatmap(
    Z, rlabs, clabs,
    title="AA×AA Z-score across simulations (clustered)",
    cbar_label="Z = mean / SD",
    diverging=True,
    outfile=out_folder+"/AAxAA_zscore.png"
)

# 5) **Difference map** between two groups (e.g., WT vs Mut)
groupA = [n for n in names if "WT" in n]
groupB = [n for n in names if "Mut" in n]
D, MA, MB = condition_difference(groupA, groupB, out_folder, box, mode="attractive")
plot_clustered_heatmap(
    D, rlabs, clabs,
    title="AA×AA difference (WT − Mut) (clustered)",
    cbar_label="Δ interaction",
    diverging=True,
    outfile=out_folder+"/AAxAA_diff_WT_minus_Mut.png"
)

# 6) **Group-collapsed 6×6** view (forces overview)
Mg, gnames = collapse_groups(Mmean)
plot_clustered_heatmap(
    Mg, gnames, gnames,
    title="Group-collapsed (6×6) AA×AA mean (clustered)",
    cbar_label="Avg interaction (by group)",
    diverging=False,
    outfile=out_folder+"/AAxAA_group_collapsed.png"
)

# 7) **Top pairs list**
tp = top_pairs(Mmean, rlabs, clabs, k=20)
print(tp.to_string(index=False))
tp.to_csv(out_folder+"/AAxAA_top_pairs.csv", index=False)
