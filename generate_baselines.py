import os, re, json, datetime, warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors


IN_DIR_REAL = "datasets/real"
OUT_DIR_SYN = "datasets/synthetic"
os.makedirs(OUT_DIR_SYN, exist_ok=True)
TUNING_LOG_CSV    = os.path.join(OUT_DIR_SYN, "_tuning_log.csv")
TUNING_TRIALS_CSV = os.path.join(OUT_DIR_SYN, "_tuning_trials.csv")


DISORDERS = [
    "depression",
    "separation_anxiety",
    "specific_phobia",
    "social_anxiety",
    "panic",
    "agoraphobia",
    "generalized_anxiety",
]
DISORDER_SET = set(DISORDERS)


# ---------------- Tuning params ----------------
EPOCH_GRID_CTGAN = [50, 100, 150, 200, 300]
EPOCH_GRID_TVAE  = [50, 100, 150, 200, 300]
TUNE_SEEDS       = [11, 23, 37]
SPLIT_RANDOM_STATE = 777
SEED             = 42           


PRIV_THR_OVERLAP = float(os.environ.get("PRIV_THR_OVERLAP", 0.01))   # ≤ 1%
PRIV_THR_SHARE_D1 = float(os.environ.get("PRIV_THR_SHARE_D1", 0.10)) # ≤ 10%
PRIV_THR_Q05 = float(os.environ.get("PRIV_THR_Q05", 1.0))            # ≥ 1


USE_NEW_METADATA = os.environ.get("SDV_USE_NEW_METADATA", "0").lower() in ("1","true","yes")
SILENCE_STM_WARN = os.environ.get("SDV_SILENCE_STM_WARN", "0").lower() in ("1","true","yes")
if SILENCE_STM_WARN:
    warnings.filterwarnings("ignore",
        message="The 'SingleTableMetadata' is deprecated",
        category=FutureWarning)

# helpers


def _encode_categorical_union(dfs,cols):
    """
    Consistently encode categorical columns across multiple dataframes.
    Returns numpy integer arrays (shape: [n_rows, p]).
    """
    cats = {
        c: sorted(pd.unique(pd.concat([d[c].astype(str) for d in dfs], ignore_index=True)))
        for c in cols
    }
    arrays = []
    for d in dfs:
        arr = np.column_stack([
            pd.Categorical(d[c].astype(str), categories=cats[c]).codes
            for c in cols
        ]).astype(np.int32)
        arrays.append(arr)
    return arrays

def _nn_hamming(a, b, n_jobs = -1):
    """
    Nearest-neighbor Hamming distance (normalized in [0,1]) from each row of a to set b.
    """
    if a.size == 0 or b.size == 0:
        return np.array([], dtype=float)
    nn = NearestNeighbors(n_neighbors=1, metric="hamming", n_jobs=n_jobs)
    nn.fit(b)
    dists, _ = nn.kneighbors(a, return_distance=True)
    return dists.ravel()

def dcr_overfit_share(df_train,df_holdout,df_syn,feature_cols,max_syn_rows = None,random_state= 0):
    """
    DCR overfitting sanity check.

    """
    
    syn = df_syn[feature_cols]
    if (max_syn_rows is not None) and (len(syn) > max_syn_rows):
        syn = syn.sample(n=max_syn_rows, random_state=random_state)

    train = df_train[feature_cols]
    hold  = df_holdout[feature_cols]

    a_train, a_hold, a_syn = _encode_categorical_union([train, hold, syn], feature_cols)

    d_train = _nn_hamming(a_syn, a_train)  
    d_hold  = _nn_hamming(a_syn, a_hold)    

    if len(d_train) == 0 or len(d_hold) == 0:
        return float("nan"), float("nan"), float("nan")

    share_closer = float(np.mean(d_train < d_hold))
    q05_delta    = float(np.quantile(d_train, 0.05) - np.quantile(d_hold,  0.05))
    mean_delta   = float(d_train.mean() - d_hold.mean())
    return share_closer, q05_delta, mean_delta

def infer_disorder_from_filename(fname):
    base = os.path.basename(fname).lower()
    if not (base.startswith("real_") and base.endswith(".csv")):
        return None
    slug = base.replace("real_", "").replace(".csv", "")
    return slug if slug in DISORDER_SET else None

def find_item_columns(df, disorder) :
    pat = re.compile(rf"^{re.escape(disorder)}_it(\d+)$", re.IGNORECASE)
    cols = [c for c in df.columns if pat.match(str(c))]
    cols.sort(key=lambda c: int(re.search(r"(\d+)$", c).group(1)))
    return cols



def to_int64(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    return out


def build_categorical_metadata(df_features: pd.DataFrame):
    """
    Build metadata for a single table, setting every column to 'categorical'.

    """
    # Try new API if requested
    if USE_NEW_METADATA:
        try:
            from sdv.metadata import Metadata as _Meta
            md = _Meta()
            # Try modern signatures
            detected = False
            if hasattr(md, "detect_table_from_dataframe"):
                try:
                    md.detect_table_from_dataframe(table_name="table", data=df_features)
                    detected = True
                except Exception:
                    detected = False
            if not detected and hasattr(md, "detect_from_dataframe"):
                try:
                    md.detect_from_dataframe(data=df_features, table_name="table")
                    detected = True
                except TypeError:
                    try:
                        md.detect_from_dataframe(df_features, table_name="table")
                        detected = True
                    except Exception:
                        pass
            if not detected and hasattr(md, "detect_from_dataframes"):
                try:
                    md.detect_from_dataframes(data={"table": df_features})
                    detected = True
                except Exception:
                    pass
            if not detected and hasattr(md, "add_table"):
                try:
                    md.add_table(name="table", data=df_features)
                    detected = True
                except Exception:
                    pass
            # Verify table exists
            try:
                tables = md.get_tables()
            except Exception:
                tables = getattr(md, "tables", None)
            has_table = bool(tables) and ("table" in (tables if isinstance(tables, (list, set, tuple)) else [tables]))
            if not has_table:
                raise RuntimeError("Metadata: no tables attached after detection attempts.")
            # Mark categorical
            for col in df_features.columns:
                try:
                    md.update_column(table_name="table", column_name=col, sdtype="categorical")
                except TypeError:
                    md.update_column("table", col, sdtype="categorical")
            return md
        except Exception as e:
            print(f"[meta] new Metadata API failed ({e}); falling back to SingleTableMetadata.")

    # Fallback: SingleTableMetadata (works across SDV versions)
    from sdv.metadata import SingleTableMetadata as _STM
    md = _STM()
    md.detect_from_dataframe(df_features)
    for col in df_features.columns:
        md.update_column(col, sdtype="categorical")
    return md

def set_seed_if_supported(synth, seed: int):
    if hasattr(synth, "set_random_state"):
        synth.set_random_state(seed)


def mean_univariate_jsd(df_real, df_syn) -> float:
    vals = []
    for c in df_real.columns:
        cats = sorted(set(df_real[c].dropna().unique()).union(df_syn[c].dropna().unique()))
        pr = df_real[c].value_counts(normalize=True).reindex(cats, fill_value=0.0).values
        ps = df_syn[c].value_counts(normalize=True).reindex(cats, fill_value=0.0).values
        vals.append(jensenshannon(pr, ps, base=2.0))
    return float(np.nanmean(vals))


def _align_encode_union(dfa, dfb):
    """Return integer-coded arrays  with shared category maps across columns."""
    cols = list(dfa.columns)
    A = np.empty((len(dfa), len(cols)), dtype=np.int32)
    B = np.empty((len(dfb), len(cols)), dtype=np.int32)
    for j, c in enumerate(cols):
        a = dfa[c].astype("object")
        b = dfb[c].astype("object")
        cats = pd.Index(a.dropna().unique()).union(pd.Index(b.dropna().unique()))
        cat_to_code = {k: i for i, k in enumerate(cats)}
        A[:, j] = a.map(cat_to_code).fillna(-1).astype(np.int32).values
        B[:, j] = b.map(cat_to_code).fillna(-1).astype(np.int32).values
    return A, B

def exact_overlap_rate(df_train, df_syn) -> float:
    """Fraction of synthetic rows that exactly match some real row."""
    real_set = set(map(tuple, df_train.astype(str).itertuples(index=False, name=None)))
    syn_tuples = list(map(tuple, df_syn.astype(str).itertuples(index=False, name=None)))
    hits = sum(1 for t in syn_tuples if t in real_set)
    return hits / max(1, len(syn_tuples))

def nn_hamming_stats(df_train, df_syn, q = 0.05):
    """Return (q-quantile of NN Hamming distance, share with d<=1)."""
    A, B = _align_encode_union(df_train, df_syn)
    nR, p = A.shape
    nS = B.shape[0]
    dmins = np.empty(nS, dtype=np.int32)
    for i in range(nS):
        diffs = (A != B[i])
        dists = diffs.sum(axis=1)
        dmins[i] = int(dists.min()) if nR > 0 else p
    qval = float(np.quantile(dmins, q)) if nS > 0 else float("nan")
    share_d1 = float((dmins <= 1).mean()) if nS > 0 else float("nan")
    return qval, share_d1


def fit_sample_sdv(method, df_features, n, epochs, seed) -> pd.DataFrame:
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
    md = build_categorical_metadata(df_features)
    if method == "ctgan":
        synth = CTGANSynthesizer(md, epochs=epochs, cuda=False, verbose=False)
    elif method == "tvae":
        synth = TVAESynthesizer(md, epochs=epochs, cuda=False, verbose=False)
    elif method == "gaussiancopula":
        synth = GaussianCopulaSynthesizer(md)
    else:
        raise ValueError(method)
    set_seed_if_supported(synth, seed)
    synth.fit(df_features)
    return synth.sample(num_rows=n)

# Privacy-aware selection
def select_epochs_privacy_aware(method, df_features):
    """
    Returns: best_epochs, best_seed, best_metrics_dict, trials_list[dicts]
    Each trial dict: epochs, seed, jsd, overlap, share_d1, d_q05
    """
    strat = df_features["sex"] if "sex" in df_features.columns else None
    df_train, df_tune = train_test_split(
        df_features, test_size=0.30, random_state=SPLIT_RANDOM_STATE, stratify=strat
    )
    grid = EPOCH_GRID_CTGAN if method == "ctgan" else EPOCH_GRID_TVAE
    trials: List[Dict[str, Any]] = []
    for seed in TUNE_SEEDS:
        for ep in grid:
            try:
                syn = fit_sample_sdv(method, df_train, len(df_tune), ep, seed)
            except Exception as e:
                trials.append(dict(epochs=ep, seed=seed, jsd=np.nan,
                                   overlap=np.nan, share_d1=np.nan, d_q05=np.nan, err=str(e)))
                continue
            jsd = mean_univariate_jsd(df_tune, syn)
            d_q05, share_d1 = nn_hamming_stats(df_train, syn, q=0.05)
            overlap = exact_overlap_rate(df_train, syn)
            trials.append(dict(epochs=ep, seed=seed, jsd=jsd,
                               overlap=overlap, share_d1=share_d1, d_q05=d_q05))
    feas = [t for t in trials
            if np.isfinite(t.get("jsd", np.nan))
            and t.get("overlap", np.inf) <= PRIV_THR_OVERLAP
            and t.get("share_d1", np.inf) <= PRIV_THR_SHARE_D1
            and t.get("d_q05", -np.inf) >= PRIV_THR_Q05]
    if feas:
        best = min(feas, key=lambda t: t["jsd"])
    else:
        finite = [t for t in trials if np.isfinite(t.get("jsd", np.nan))]
        if not finite:
            return (grid[-1], SEED,
                    dict(epochs=grid[-1], seed=SEED, jsd=float("nan"),
                         overlap=float("nan"), share_d1=float("nan"), d_q05=float("nan"),
                         selected_rule="fallback_none_feasible"),
                    trials)
        best = min(finite, key=lambda t: (t.get("overlap", np.inf),
                                          t.get("share_d1", np.inf),
                                          -t.get("d_q05", -np.inf),
                                          t["jsd"]))
        best["selected_rule"] = "lexicographic_fallback"
    best.setdefault("selected_rule", "privacy_thresholds_then_min_jsd")
    return best["epochs"], best["seed"], best, trials


def random_with_empirical_marginals(df_real, disorder,item_cols, n, seed) :
    rng = np.random.default_rng(seed)
    cols = []
    if "sex" in df_real.columns: cols.append("sex")
    if "age" in df_real.columns: cols.append("age")
    cols += item_cols
    out = pd.DataFrame(index=range(n), columns=cols)

    # sex empirical pmf
    if "sex" in out.columns:
        vc = pd.to_numeric(df_real["sex"], errors="coerce").dropna().astype(int).value_counts(normalize=True).sort_index()
        out["sex"] = rng.choice(vc.index.to_list(), size=n, replace=True, p=vc.values)

    # age  empirical pmf (discrete)
    if "age" in out.columns:
        vc = pd.to_numeric(df_real["age"], errors="coerce").dropna().astype(int).value_counts(normalize=True).sort_index()
        out["age"] = rng.choice(vc.index.to_list(), size=n, replace=True, p=vc.values)

    # items  uniform 
    U = 3 if disorder == "depression" else 4
    for c in item_cols:
        out[c] = rng.integers(0, U + 1, size=n)

    return to_int64(out)

#utils
def append_row_csv(path, row, field_order):
    df_row = pd.DataFrame([{k: row.get(k, "") for k in field_order}])
    write_header = (not os.path.exists(path))
    df_row.to_csv(path, mode="a", header=write_header, index=False)

def process_one_real_csv(path):
    base = os.path.basename(path)
    disorder = infer_disorder_from_filename(base)
    if not disorder:
        print(f"[skip] {base}: cannot infer disorder")
        return

    df = pd.read_csv(path)
    items = find_item_columns(df, disorder)

    feat_cols = []
    if "sex" in df.columns: feat_cols.append("sex")
    if "age" in df.columns: feat_cols.append("age")
    feat_cols += items
    feat_cols = [c for c in feat_cols if c in df.columns]

    if not feat_cols or not items:
        print(f"[skip] {base}: missing features/items")
        return

    df_features = df[feat_cols].copy()
    n = len(df_features)
    ts = datetime.datetime.utcnow().isoformat() + "Z"

    # CTGAN privacy-aware tuning
    try:
        best_ep, best_seed, best_metrics, trials = select_epochs_privacy_aware("ctgan", df_features)
        s = fit_sample_sdv("ctgan", df_features, n, best_ep, best_seed)
        s = to_int64(s)
        out_csv = os.path.join(OUT_DIR_SYN, f"synthetic_{disorder}_ctgan.csv")
        s.to_csv(out_csv, index=False)
        meta = {
            "disorder": disorder, "method": "ctgan",
            "best_epochs": best_ep, "best_seed": best_seed,
            "best_tune_jsd": best_metrics.get("jsd"),
            "best_tune_overlap": best_metrics.get("overlap"),
            "best_tune_share_d1": best_metrics.get("share_d1"),
            "best_tune_d_q05": best_metrics.get("d_q05"),
            "selection_rule": best_metrics.get("selected_rule", ""),
            "epoch_grid": EPOCH_GRID_CTGAN, "tune_seeds": TUNE_SEEDS,
            "privacy_thresholds": {
                "overlap": PRIV_THR_OVERLAP, "share_d1": PRIV_THR_SHARE_D1, "d_q05": PRIV_THR_Q05
            },
            "split_random_state": SPLIT_RANDOM_STATE, "n_full": n
        }
        with open(out_csv.replace(".csv", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        append_row_csv(
            TUNING_LOG_CSV,
            {
                "timestamp_utc": ts, "disorder": disorder, "method": "ctgan",
                "best_epochs": best_ep, "best_seed": best_seed,
                "best_jsd_tune": best_metrics.get("jsd"),
                "best_overlap_tune": best_metrics.get("overlap"),
                "best_share_d1_tune": best_metrics.get("share_d1"),
                "best_d_q05_tune": best_metrics.get("d_q05"),
                "selection_rule": best_metrics.get("selected_rule", ""),
                "epoch_grid": "|".join(map(str, EPOCH_GRID_CTGAN)),
                "tune_seeds": "|".join(map(str, TUNE_SEEDS)),
                "split_random_state": SPLIT_RANDOM_STATE,
                "n_train": int(np.round(0.9*n)), "n_tune": n - int(np.round(0.9*n))
            },
            ["timestamp_utc","disorder","method","best_epochs","best_seed",
             "best_jsd_tune","best_overlap_tune","best_share_d1_tune","best_d_q05_tune",
             "selection_rule","epoch_grid","tune_seeds","split_random_state","n_train","n_tune"]
        )
        for t in trials:
            append_row_csv(
                TUNING_TRIALS_CSV,
                {"timestamp_utc": ts, "disorder": disorder, "method": "ctgan", **t},
                ["timestamp_utc","disorder","method","epochs","seed","jsd","overlap","share_d1","d_q05","err"]
            )
        print(f"[ok][CTGAN] {base} ep={best_ep} seed={best_seed} jsd={best_metrics.get('jsd'):.4f} "
              f"ov={best_metrics.get('overlap'):.4f} d05={best_metrics.get('d_q05'):.2f} s<=1={best_metrics.get('share_d1'):.4f}")
    except Exception as e:
        print(f"[warn] CTGAN failed for {base}: {e}")

    # TVAE privacy-aware tuning
    try:
        best_ep, best_seed, best_metrics, trials = select_epochs_privacy_aware("tvae", df_features)
        s = fit_sample_sdv("tvae", df_features, n, best_ep, best_seed)
        s = to_int64(s)
        out_csv = os.path.join(OUT_DIR_SYN, f"synthetic_{disorder}_tvae.csv")
        s.to_csv(out_csv, index=False)
        meta = {
            "disorder": disorder, "method": "tvae",
            "best_epochs": best_ep, "best_seed": best_seed,
            "best_tune_jsd": best_metrics.get("jsd"),
            "best_tune_overlap": best_metrics.get("overlap"),
            "best_tune_share_d1": best_metrics.get("share_d1"),
            "best_tune_d_q05": best_metrics.get("d_q05"),
            "selection_rule": best_metrics.get("selected_rule", ""),
            "epoch_grid": EPOCH_GRID_TVAE, "tune_seeds": TUNE_SEEDS,
            "privacy_thresholds": {
                "overlap": PRIV_THR_OVERLAP, "share_d1": PRIV_THR_SHARE_D1, "d_q05": PRIV_THR_Q05
            },
            "split_random_state": SPLIT_RANDOM_STATE, "n_full": n
        }
        with open(out_csv.replace(".csv", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        append_row_csv(
            TUNING_LOG_CSV,
            {
                "timestamp_utc": ts, "disorder": disorder, "method": "tvae",
                "best_epochs": best_ep, "best_seed": best_seed,
                "best_jsd_tune": best_metrics.get("jsd"),
                "best_overlap_tune": best_metrics.get("overlap"),
                "best_share_d1_tune": best_metrics.get("share_d1"),
                "best_d_q05_tune": best_metrics.get("d_q05"),
                "selection_rule": best_metrics.get("selected_rule", ""),
                "epoch_grid": "|".join(map(str, EPOCH_GRID_TVAE)),
                "tune_seeds": "|".join(map(str, TUNE_SEEDS)),
                "split_random_state": SPLIT_RANDOM_STATE,
                "n_train": int(np.round(0.9*n)), "n_tune": n - int(np.round(0.9*n))
            },
            ["timestamp_utc","disorder","method","best_epochs","best_seed",
             "best_jsd_tune","best_overlap_tune","best_share_d1_tune","best_d_q05_tune",
             "selection_rule","epoch_grid","tune_seeds","split_random_state","n_train","n_tune"]
        )
        for t in trials:
            append_row_csv(
                TUNING_TRIALS_CSV,
                {"timestamp_utc": ts, "disorder": disorder, "method": "tvae", **t},
                ["timestamp_utc","disorder","method","epochs","seed","jsd","overlap","share_d1","d_q05","err"]
            )
        print(f"[ok][TVAE] {base} ep={best_ep} seed={best_seed} jsd={best_metrics.get('jsd'):.4f} "
              f"ov={best_metrics.get('overlap'):.4f} d05={best_metrics.get('d_q05'):.2f} s<=1={best_metrics.get('share_d1'):.4f}")
    except Exception as e:
        print(f"[warn] TVAE failed for {base}: {e}")



    # Random with empirical marginals (sex/age)
    try:
        s = random_with_empirical_marginals(df, disorder, items, n, SEED)
        out_csv = os.path.join(OUT_DIR_SYN, f"synthetic_{disorder}_random.csv")
        s.to_csv(out_csv, index=False)
        with open(out_csv.replace(".csv", "_meta.json"), "w") as f:
            json.dump({"disorder": disorder, "method": "random",
                       "sex_age": "empirical_marginals", "n_full": n}, f, indent=2)
        print(f"[ok][Random] {base}")
    except Exception as e:
        print(f"[warn] Random failed for {base}: {e}")

def main():
    files = [os.path.join(IN_DIR_REAL, f) for f in sorted(os.listdir(IN_DIR_REAL))
             if f.startswith("real_") and f.endswith(".csv")]
    if not files:
        print(f"[gen] No CSVs found in {IN_DIR_REAL}")
        return
    for p in files:
        process_one_real_csv(p)

if __name__ == "__main__":
    main()
