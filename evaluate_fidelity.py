# -------- Cap BLAS/OMP threads (helps on multi-core / Apple Silicon) --------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import pandas as pd

# Paths & Config 
REAL_DIR   = Path("datasets/real")
SYN_DIR    = Path("datasets/synthetic")
OUT_DIR    = Path("results/fidelity")
OUT_DIR.mkdir(parents=True, exist_ok=True)


DISORDERS = [
    "separation_anxiety","specific_phobia",
    "social_anxiety","panic","agoraphobia","generalized_anxiety",
]
LLM_METHODS = ["dsm5", "icd10", "dsm5+icd10", "none"]
BASELINE_METHODS = ["random", "ctgan", "tvae"]
ALL_METHODS = LLM_METHODS + BASELINE_METHODS

# Bootstrap
B_BOOT = 5000
SEED = 42

# JSD smoothing
ALPHA_DIRICHLET = 0.5
NA_TOKEN = "__NA__"

# Helpers
def item_cols(df, disorder):
    
    pref = f"{disorder}_it"
    cols = [c for c in df.columns if str(c).startswith(pref)]
    # sort numerically 
    def idx(c: str) -> int:
        return int(str(c).split(pref)[1])
        
    cols.sort(key=idx)
    return cols

def select_features(df, disorder):
    cols = set(df.columns)
    feats = []
    if "sex" in cols: feats.append("sex")
    if "age" in cols: feats.append("age")
    feats += item_cols(df, disorder)
    feats = [c for c in feats if c != "target"]
    return feats

def harmonize_features(dfA, dfB, disorder):
    fA = set(select_features(dfA, disorder))
    fB = set(select_features(dfB, disorder))
    feats = sorted(fA.intersection(fB), key=lambda c: (c not in ("sex","age"), c))
    return feats

def as_cat(s):
    s = s.astype("object")
    return s.where(~s.isna(), NA_TOKEN)

# Metrics 
def harmonized_counts(sr, ss) :
    r = as_cat(sr); s = as_cat(ss)
    levels = pd.Index(pd.unique(pd.concat([r, s], ignore_index=True))).tolist()
    nR = r.value_counts().reindex(levels, fill_value=0).to_numpy(float)
    nS = s.value_counts().reindex(levels, fill_value=0).to_numpy(float)
    return nR, nS

def jsd_feature(sr, ss, alpha = ALPHA_DIRICHLET) :
    nR, nS = harmonized_counts(sr, ss)
    k = max(1, len(nR))
    PR = (nR + alpha) / (nR.sum() + alpha * k)
    PS = (nS + alpha) / (nS.sum() + alpha * k)
    M  = 0.5 * (PR + PS)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_RM = np.where(PR > 0, PR * (np.log(PR) - np.log(M)), 0.0).sum()
        kl_SM = np.where(PS > 0, PS * (np.log(PS) - np.log(M)), 0.0).sum()
    return float(0.5 * (kl_RM + kl_SM) / math.log(2.0))  # in [0,1]

def JSD(real, synth, feats):
    if not feats:
        return float("nan")
    vals = [jsd_feature(real[c], synth[c]) for c in feats]
    return float(np.mean(vals)) if vals else float("nan")

def cramers_v_corr(tab):
    n = tab.sum()
    if n <= 1:
        return 0.0
    row = tab.sum(axis=1, keepdims=True)
    col = tab.sum(axis=0, keepdims=True)
    exp = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0).sum()
    phi2 = chi2 / n
    r, c = tab.shape
    phi2c = max(0.0, phi2 - ((r - 1) * (c - 1)) / max(1, (n - 1)))
    rcmin = max(1, min(r - 1, c - 1))
    return float(min(1.0, math.sqrt(phi2c / rcmin)))

def MAE_V(real, synth, feats):
    p = len(feats)
    if p < 2:
        return float("nan")
    diffs = []
    for i in range(p):
        for j in range(i + 1, p):
            a, b = feats[i], feats[j]
            R = pd.crosstab(as_cat(real[a]), as_cat(real[b]))
            S = pd.crosstab(as_cat(synth[a]), as_cat(synth[b]))
            idx = R.index.union(S.index)
            col = R.columns.union(S.columns)
            rtab = R.reindex(index=idx, columns=col, fill_value=0).to_numpy(float)
            stab = S.reindex(index=idx, columns=col, fill_value=0).to_numpy(float)
            diffs.append(abs(cramers_v_corr(rtab) - cramers_v_corr(stab)))
    return float(np.mean(diffs)) if diffs else float("nan")

def encode_categorical_block(real, synth, feats):
   
    cats = {c: pd.Index(pd.unique(pd.concat([as_cat(real[c]), as_cat(synth[c])], ignore_index=True))) for c in feats}
    def to_codes(df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in feats:
            cols.append(pd.Categorical(as_cat(df[c]), categories=cats[c]).codes.astype(np.int32))
        return np.column_stack(cols)
    return to_codes(real[feats]), to_codes(synth[feats])

def ED2_V(
    real,
    synth,
    feats,
    n_pairs_cross = 200_000,
    n_pairs_within = 200_000,
    normalize= True,
    random_state = 0):
    """
    Energy Distance squared with V-statistic, using Hamming distance:
        ED_V^2 = 2 E[d(X,Y)] - E[d(X,X')] - E[d(Y,Y')]
    expectation via Monte-Carlo pairs with replacement*.
    """
    if not feats:
        return float("nan")
    Xr, Xs = encode_categorical_block(real, synth, feats)
    n, m = Xr.shape[0], Xs.shape[0]
    if n == 0 or m == 0:
        return float("nan")
    rng = np.random.default_rng(random_state)
    p = Xr.shape[1]
    denom = float(p) if normalize else 1.0

    # Cross E[d(X,Y)]
    k_xy = min(n * m, n_pairs_cross)
    i_xy = rng.integers(0, n, size=k_xy)
    j_xy = rng.integers(0, m, size=k_xy)
    e_xy = np.mean((Xr[i_xy] != Xs[j_xy]).sum(axis=1) / denom)

    # Within E[d(X,X')] (V-stat includes i=j)
    k_xx = min(n * n, n_pairs_within)
    i_xx = rng.integers(0, n, size=k_xx)
    j_xx = rng.integers(0, n, size=k_xx)
    e_xx = np.mean((Xr[i_xx] != Xr[j_xx]).sum(axis=1) / denom)

    # Within E[d(Y,Y')] (V-stat includes i=j)
    k_yy = min(m * m, n_pairs_within)
    i_yy = rng.integers(0, m, size=k_yy)
    j_yy = rng.integers(0, m, size=k_yy)
    e_yy = np.mean((Xs[i_yy] != Xs[j_yy]).sum(axis=1) / denom)

    return float(2.0 * e_xy - e_xx - e_yy)



def rng_for(seed, *keys):

    ss = np.random.SeedSequence([int(seed), *map(int, keys)])
    return np.random.default_rng(ss)

def sample_match(df, n, rng):
    idx = rng.integers(0, len(df), size=n)
    return df.iloc[idx].reset_index(drop=True)

def summarize(arr) :
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return float(a.mean()), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def main():
    rows_raw = []
    rows_sum = []

    for disorder in DISORDERS:
        print(f"Processing disorder: {disorder}")
        real_path = REAL_DIR / f"real_{disorder}.csv"

        dfR_all = pd.read_csv(real_path)

        for method in ALL_METHODS:
            print(f"  Method: {method}")
            synth_path = SYN_DIR / f"synthetic_{disorder}_{method}.csv"


            dfS_all = pd.read_csv(synth_path)
            feats = harmonize_features(dfR_all, dfS_all, disorder)


            dfR = dfR_all[feats].reset_index(drop=True)
            dfS = dfS_all[feats].reset_index(drop=True)
            n = len(dfR)
            rng = rng_for(SEED, len(feats), n)

            jsd_b = np.empty(B_BOOT, float)
            mae_b = np.empty(B_BOOT, float)
            ed2_b = np.empty(B_BOOT, float)

            for b in range(B_BOOT):
                rb = sample_match(dfR, n, rng)
                sb = sample_match(dfS, n, rng)
                jsd_b[b] = JSD(rb, sb, feats)
                mae_b[b] = MAE_V(rb, sb, feats)
                ed2_b[b] = ED2_V(rb, sb, feats, random_state=int(SEED + b))

            for metric, arr in (("JSD", jsd_b), ("MAE_V", mae_b), ("ED2", ed2_b)):
                mu, lo, hi = summarize(arr)
                rows_sum.append({
                    "disorder": disorder, "method": method, "metric": metric,
                    "mean": mu, "ci_lo": lo, "ci_hi": hi, "n_boot": int(len(arr)),
                    "n_real": int(len(dfR)), "n_synth": int(len(dfS)),
                })
                for i, v in enumerate(arr):
                    rows_raw.append({
                        "disorder": disorder, "method": method, "metric": metric,
                        "bootstrap": i, "value": float(v),
                        "n_real": int(len(dfR)), "n_synth": int(len(dfS)),
                    })

    pd.DataFrame(rows_sum).to_csv(OUT_DIR / "fidelity_summary.csv", index=False)
    pd.DataFrame(rows_raw).to_csv(OUT_DIR / "fidelity_bootstrap_raw.csv", index=False)
    print(f"wrote: {OUT_DIR/'fidelity_summary.csv'}")
    print(f"wrote: {OUT_DIR/'fidelity_bootstrap_raw.csv'}")

if __name__ == "__main__":
    main()
