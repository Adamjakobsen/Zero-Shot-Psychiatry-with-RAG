import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ANXIETY_ONLY_EXCLUDE = {"depression"}
KNOWLEDGE = ("icd10", "dsm5", "dsm5+icd10")
METRICS_CANON = ("JSD", "MAE_V", "ED2")  # 'PCD' in raw is renamed to 'MAE_V'

DISP_LABEL = {
    "agoraphobia": "Agoraphobia",
    "generalized_anxiety": "Generalized anxiety",
    "panic": "Panic",
    "separation_anxiety": "Separation anxiety",
    "social_anxiety": "Social anxiety",
    "specific_phobia": "Specific phobia",
}

def load_bootstrap(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["disorder"] = df["disorder"].str.lower()
    df["method"]   = df["method"].str.lower()
    df["metric"]   = df["metric"].str.upper()
    df = df[~df["disorder"].isin(ANXIETY_ONLY_EXCLUDE)].copy()
    # Rename PCD -> MAE_V (NO transformation)
    df.loc[df["metric"] == "PCD", "metric"] = "MAE_V"
    # Keep only the metrics we report
    df = df[df["metric"].isin(METRICS_CANON)].copy()
    return df

def paired_delta(none_s, meth_s) :
    """Return per-bootstrap delta = none - method on aligned bootstrap indices."""
    a = none_s.dropna()
    b = meth_s.dropna()
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return np.array([], dtype=float)
    return a.loc[idx].to_numpy(dtype=float) - b.loc[idx].to_numpy(dtype=float)

def summarize(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    mu = float(np.mean(a))
    lo, hi = np.quantile(a, [0.025, 0.975])
    return (mu, float(lo), float(hi))

def bold_mask(values, tol):
    """Bold the row-wise maximum (ties allowed)."""
    arr = np.array([(-np.inf if not np.isfinite(v) else v) for v in values], float)
    if arr.size == 0:
        return [False]*len(values)
    m = np.nanmax(arr)
    return [(abs(v - m) <= tol) if np.isfinite(v) else False for v in values]

def fmt_cell(mu: float, lo: float, hi: float, bold: bool) -> str:
    if not np.isfinite(mu):
        return ""
    s = f"{mu:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"\\textbf{{{s}}}" if bold else s

def build_delta(df) :
    rows = []
    for dis, gD in df.groupby("disorder"):
        if "none" not in set(gD["method"]):
            continue
        disp = DISP_LABEL.get(dis, dis.replace("_", " "))
        gD = gD.set_index(["metric", "method", "bootstrap"]).sort_index()

        row = {"Disorder": disp}
        for met in METRICS_CANON:
            if met not in gD.index.get_level_values("metric"):
                for meth in KNOWLEDGE:
                    row[f"{met}__{meth}__mean"] = np.nan
                    row[f"{met}__{meth}__lo"]   = np.nan
                    row[f"{met}__{meth}__hi"]   = np.nan
                    row[f"{met}__{meth}__bold"] = False
                continue

            none_b = gD.xs((met, "none"), level=("metric","method"))["value"]
            means = []
            stats = {}
            for meth in KNOWLEDGE:
                key = (met, meth)
                if key in gD.index:
                    m_b = gD.xs(key, level=("metric","method"))["value"]
                    d = paired_delta(none_b, m_b)
                    mu, lo, hi = summarize(d)
                else:
                    mu = lo = hi = np.nan
                stats[meth] = (mu, lo, hi)
                means.append(mu)

            mask = bold_mask(means, tol=0.0)
            for j, meth in enumerate(KNOWLEDGE):
                mu, lo, hi = stats[meth]
                row[f"{met}__{meth}__mean"] = mu
                row[f"{met}__{meth}__lo"]   = lo
                row[f"{met}__{meth}__hi"]   = hi
                row[f"{met}__{meth}__bold"] = bool(mask[j])

        rows.append(row)

    dfw = pd.DataFrame(rows).sort_values("Disorder").reset_index(drop=True)
    return dfw

def write_csv(dfw, out_csv):
    dfw.to_csv(out_csv, index=False)

def write_latex(dfw, out_tex):

    order = ["Agoraphobia", "Generalized anxiety", "Panic",
             "Separation anxiety", "Social anxiety", "Specific phobia"]
    dfw = dfw.copy()
    
    def _norm(x): return x.strip()
    dfw["Disorder"] = dfw["Disorder"].map(_norm)
    
    disorders = [d for d in order if d in set(dfw["Disorder"])] + \
                [d for d in dfw["Disorder"].tolist() if d not in order]

    # Helper to fetch a cell tuple (mu, lo, hi) for (disorder, metric, method)
    def get_triplet(dis, met, meth):
        r = dfw[dfw["Disorder"] == dis]
        if r.empty:
            return (np.nan, np.nan, np.nan)
        mu = r[f"{met}__{meth}__mean"].values[0] if f"{met}__{meth}__mean" in r.columns else np.nan
        lo = r[f"{met}__{meth}__lo"].values[0]   if f"{met}__{meth}__lo"   in r.columns else np.nan
        hi = r[f"{met}__{meth}__hi"].values[0]   if f"{met}__{meth}__hi"   in r.columns else np.nan
        return (mu, lo, hi)

    
    def fmt(mu, lo, hi, bold=False):
        if not np.isfinite(mu):
            return ""
        s = f"{mu:.3f} [{lo:.3f}, {hi:.3f}]"
        return f"\\textbf{{{s}}}" if bold else s

    
    methods = [("icd10",  "\\textit{\\ICDten}"),
               ("dsm5",   "\\textit{\\DSMV}"),
               ("dsm5+icd10", "\\textit{\\DualKB}")]

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\caption{Improvements over \\textit{None} with paired 95\\% CIs. "
                 "Best per metric within each disorder is \\textbf{bold}.}")
    lines.append("\\label{tab:delta_fidelity_narrow}")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("\\rotatebox{70}{\\textbf{Disorder}} & "
                 "\\rotatebox{70}{\\textbf{Method}} & "
                 "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{$\\Delta$JSD} $\\uparrow$}} & "
                 "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{$\\Delta$MAE$_V$} $\\uparrow$}} & "
                 "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{$\\Delta$ED$^2$} $\\uparrow$}} \\\\")
    lines.append("\\midrule")

    for d in disorders:
        # Collect means for bolding within this disorder (per metric across methods)
        jsd_vals = [get_triplet(d, "JSD", m)[0]   for m, _ in methods]
        msev_vals= [get_triplet(d, "MAE_V", m)[0] for m, _ in methods]
        ed2_vals = [get_triplet(d, "ED2", m)[0]   for m, _ in methods]

        # Determine bold masks 
        def bold_mask(vals):
            arr = np.array([(-np.inf if not np.isfinite(v) else v) for v in vals], float)
            if arr.size == 0:
                return [False]*len(vals)
            m = np.nanmax(arr)
            return [np.isfinite(v) and (v >= m) for v in vals]

        bJ, bM, bE = bold_mask(jsd_vals), bold_mask(msev_vals), bold_mask(ed2_vals)

        # Emit three rows (one per KB method)
        first = True
        for idx, (m_slug, m_disp) in enumerate(methods):
            muJ, loJ, hiJ = get_triplet(d, "JSD",   m_slug)
            muM, loM, hiM = get_triplet(d, "MAE_V", m_slug)
            muE, loE, hiE = get_triplet(d, "ED2",   m_slug)

            if first:
                lead = f"\\multirow{{3}}{{*}}{{\\rotatebox{{90}}{{{d}}}}}"
                first = False
            else:
                lead = " "

            line = (f"{lead} & {m_disp} & "
                    f"{fmt(muJ, loJ, hiJ, bJ[idx])} & "
                    f"{fmt(muM, loM, hiM, bM[idx])} & "
                    f"{fmt(muE, loE, hiE, bE[idx])} \\\\")
            lines.append(line)
        lines.append("\\midrule")

    # Replace the last \midrule with \bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("./results/fidelity/fidelity_bootstrap_raw.csv"),
                    help="Path to fidelity_bootstrap_raw.csv")
    ap.add_argument("--outdir", type=Path, default=Path("./results/fidelity"),
                    help="Directory for outputs (CSV and TEX)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = load_bootstrap(args.raw)
    dfw = build_delta(raw)

    out_csv = args.outdir / "delta_vs_none_bootstrap.csv"
    out_tex = args.outdir / "delta_vs_none_bootstrap.tex"
    write_csv(dfw, out_csv)
    write_latex(dfw, out_tex)

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_tex}")

if __name__ == "__main__":
    main()



