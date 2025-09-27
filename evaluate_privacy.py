# -------- Cap BLAS/OMP threads (helps on multi-core / Apple Silicon) --------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import re
from collections import Counter

# Paths & Config
REAL_DIR = Path("datasets/real")
SYN_DIR  = Path("datasets/synthetic")
OUT_DIR  = Path("results/privacy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed experiment scope 
DISORDERS = [
    "separation_anxiety","specific_phobia",
    "social_anxiety","panic","agoraphobia","generalized_anxiety",
]
LLM_METHODS      = ["dsm5", "icd10", "dsm5+icd10", "none"]
BASELINE_METHODS = ["random", "ctgan", "tvae"]
ALL_METHODS = LLM_METHODS + BASELINE_METHODS

NA_TOKEN = "__NA__"
SEED = 42
rng = np.random.default_rng(SEED)

# Helper functions
def item_cols(df, disorder) :
    p = re.compile(rf"^{re.escape(disorder)}_it(\d+)$", re.IGNORECASE)
    cols = [c for c in df.columns if p.match(str(c))]
    cols.sort(key=lambda c: int(re.search(r"(\d+)$", c).group(1)))
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

def encode_to_int_codes(dfR, dfS, feats):
    cats = {c: pd.Index(pd.unique(pd.concat([as_cat(dfR[c]), as_cat(dfS[c])], ignore_index=True))) for c in feats}
    def _to(df):
        cols = []
        for c in feats:
            cols.append(pd.Categorical(as_cat(df[c]), categories=cats[c]).codes.astype(np.int32))
        return np.column_stack(cols)
    return _to(dfR[feats]), _to(dfS[feats])

def exact_overlap_rate(R, S, feats):
    # count exact matches of S rows in R on "feats"
    tuplesR = pd.MultiIndex.from_frame(R[feats].apply(as_cat)).value_counts()
    # For each S row, does it exist in R?
    exists = R[feats].apply(as_cat).merge(
        S[feats].apply(as_cat), how="right", indicator=True
    )["_merge"].eq("both").to_numpy()
    return float(np.mean(exists)) if len(exists) else np.nan

def nn_q05_and_share_le1(R_codes, S_codes):
    """
    Return (q05 of normalized NN Hamming distance, share of rows with normalized d <= 1/p).
    """
    nR, p = R_codes.shape
    nS = S_codes.shape[0]
    if nR == 0 or nS == 0 or p == 0:
        return (np.nan, np.nan)
    # chunking
    TARGET_MAX_BYTES = 200 * 1024 * 1024
    bytes_per_bool = 1
    chunk = max(1, int(TARGET_MAX_BYTES // max(1, (nR * p * bytes_per_bool))))
    chunk = min(chunk, nS)

    mins = np.empty(nS, dtype=np.int32)
    start = 0
    while start < nS:
        end = min(nS, start + chunk)
        Sc = S_codes[start:end]  # (c, p)
        diffs = (Sc[:, None, :] != R_codes[None, :, :])  
        dh = diffs.sum(axis=2)                           
        mins[start:end] = dh.min(axis=1)                 
        start = end

    dnorm = mins / p
    q05 = float(np.quantile(dnorm, 0.05))
    share_le_1_over_p = float(np.mean(dnorm <= (1.0/p)))
    return q05, share_le_1_over_p

def risk_k_map(dfR, dfS):
    """
    Risk_k-map average over synthetic rows with QI = {sex, age}. 
    """
    qi_cols = []
    if "sex" in dfR.columns and "sex" in dfS.columns: qi_cols.append("sex")
    if "age" in dfR.columns and "age" in dfS.columns: qi_cols.append("age")
    if not qi_cols or len(dfS)==0: 
        return float("nan")

    Rq = dfR[qi_cols].copy()
    Sq = dfS[qi_cols].copy()
    # coerce
    if "age" in qi_cols:
        for Q in (Rq, Sq):
            Q["age"] = pd.to_numeric(Q["age"], errors="coerce").astype("Int64")
            Q["age"] = Q["age"].astype("object").where(~Q["age"].isna(), NA_TOKEN)

    keyR = [tuple(x) for x in Rq.apply(as_cat, axis=0).itertuples(index=False, name=None)]
    keyS = [tuple(x) for x in Sq.apply(as_cat, axis=0).itertuples(index=False, name=None)]
    cntR = Counter(keyR)
    risks = []
    for k in keyS:
        c = cntR.get(k, 0)
        risks.append(0.0 if c <= 0 else (1.0 / c))
    return float(np.mean(risks)) if risks else float("nan")

def generate_latex_table(df) :
    """Generates a LaTeX table string from the privacy metrics ."""
    
    
    df = df.copy()

    # Define mappings for disorders and methods
    disorder_map = {
        "agoraphobia": "Agoraphobia",
        "generalized_anxiety": "\\parbox[b]{1.35cm}{\\centering Generalized\\\\Anxiety}",
        "panic": "Panic",
        "separation_anxiety": "\\parbox[b]{1.35cm}{\\centering Separation\\\\Anxiety}",
        "social_anxiety": "Social Anxiety",
        "specific_phobia": "\\parbox[b]{1.35cm}{\\centering Specific\\\\Phobia}",
    }
    method_map = {
        "random": "Random",
        "ctgan": "CTGAN",
        "icd10": "\\textit{\\ICDten}",
        "dsm5+icd10": "\\textit{\\DualKB}",
        "dsm5": "\\textit{\\DSMV}",
        "none": "\\textit{\\NoKB}",
        "tvae": "TVAE",
    }
    
    df["disorder_tex"] = df["disorder"].map(disorder_map)
    df["method_tex"] = df["method"].map(method_map)
    
    # Define which metrics are better lower (min) or higher (max)
    metrics = {
        "ExactOverlap": "min",
        "dNN": "max",
        "Share": "min",
        "kmap": "min",
    }
    
    # Find the best value for each metric within each disorder group
    for metric, op in metrics.items():
        if df[metric].notna().any():
            best_val = df.groupby("disorder")[metric].transform(op)
            df[f"{metric}_best"] = df[metric] == best_val
        else:
            df[f"{metric}_best"] = False


    # Set method order for the table
    method_order = [
        "Random", "CTGAN", "\\textit{\\ICDten}", "\\textit{\\DualKB}", 
        "\\textit{\\DSMV}", "\\textit{\\NoKB}", "TVAE"
    ]
    df['method_order'] = pd.Categorical(df['method_tex'], categories=method_order, ordered=True)
    df = df.sort_values(['disorder', 'method_order'])

    # --- String Building ---
    lines = []
    
    # Preamble
    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\caption{Privacy metrics. Best per metric within each disorder is \\textbf{bold}.}")
    lines.append("\\label{tab:privacy}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    
    # Header
    lines.append(
        "\\rotatebox{30}{\\textbf{Disorder}} & "
        "\\rotatebox{30}{\\textbf{Method}} & "
        "\\rotatebox{30}{\\parbox[b]{1.35cm}{\\centering Exact Overlap $\\downarrow$}} & "
        "\\rotatebox{30}{\\parbox[b]{1.8cm}{\\centering $q_{0.05}\\{d_{\\mathrm{NN}}\\}$ $\\uparrow$}} & "
        "\\rotatebox{30}{\\parbox[b]{1.9cm}{\\centering Share($d_{\\mathrm{norm}}\\!\\le\\!1/p$) $\\downarrow$}} & "
        "\\rotatebox{30}{\\parbox[b]{1.35cm}{\\centering $\\mathrm{Risk}_{\\text{k-map}}$ $\\downarrow$}} \\\\"
    )
    lines.append("\\midrule")

    # Table Body
    disorder_groups = df.groupby("disorder", sort=False)
    num_groups = len(disorder_groups)
    for i, (disorder_name, group) in enumerate(disorder_groups):
        num_rows_in_group = len(group)
        for j, row in enumerate(group.itertuples()):
            row_cells = []
            
            # Disorder column (only on the first row of the group)
            if j == 0:
                disorder_tex_name = row.disorder_tex
                row_cells.append(f"\\multirow{{{num_rows_in_group}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{disorder_tex_name}}}}}")
            else:
                # *** FIX: Add an empty placeholder for subsequent rows to ensure alignment ***
                row_cells.append("")
            
            # Method column
            row_cells.append(row.method_tex)
            
            # Metric columns
            for metric in metrics.keys():
                val = getattr(row, metric)
                is_best = getattr(row, f"{metric}_best")
                
                # Format to 3 decimal places
                formatted_val = f"{val:.3f}"
                
                # Bold if best
                if is_best:
                    formatted_val = f"\\textbf{{{formatted_val}}}"
                
                row_cells.append(formatted_val)

            # Join cells and add to lines
            lines.append(" & ".join(row_cells) + " \\\\")

        # Add midrule between groups
        if i < num_groups - 1:
            lines.append("\\midrule")

    # Footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    rows = []
    for disorder in DISORDERS:
        real_path = REAL_DIR / f"real_{disorder}.csv"
        if not real_path.exists():
            continue
        dfR_all = pd.read_csv(real_path)

        for method in ALL_METHODS:
            synth_path = SYN_DIR / f"synthetic_{disorder}_{method}.csv"
            if not synth_path.exists():
                continue
            dfS_all = pd.read_csv(synth_path)

            feats = harmonize_features(dfR_all, dfS_all, disorder)
            dfR = dfR_all[feats].reset_index(drop=True)
            dfS = dfS_all[feats].reset_index(drop=True)

            eo = exact_overlap_rate(dfR, dfS, feats)

            R_codes, S_codes = encode_to_int_codes(dfR, dfS, feats)
            q05, share_le_1_over_p = nn_q05_and_share_le1(R_codes, S_codes)

            risk = risk_k_map(dfR_all, dfS_all)  

            rows.append({
                "disorder": disorder,
                "method": method,
                "ExactOverlap": eo,
                "dNN": q05,
                "Share": share_le_1_over_p,
                "kmap": risk,
            })

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "privacy_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")
    # Generate and save LaTeX table
    if not out.empty:
        latex_string = generate_latex_table(out)
        latex_path = OUT_DIR / "privacy_results.tex"
        with open(latex_path, "w") as f:
            f.write(latex_string)
        print(f"Wrote LaTeX table to: {latex_path}")
        print("\n--- LaTeX Table ---")
        print(latex_string)
        print("--- End LaTeX Table ---\n")

if __name__ == "__main__":
    main()
