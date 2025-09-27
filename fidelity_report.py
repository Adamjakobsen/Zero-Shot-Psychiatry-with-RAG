#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path

IN_DIR  = Path("results/fidelity")
OUT_DIR = IN_DIR

DISP_ORDER = [
    ("agoraphobia", "Agoraphobia"),
    ("generalized_anxiety", "Generalized Anxiety"),
    ("panic", "Panic"),
    ("separation_anxiety", "Separation Anxiety"),
    ("social_anxiety", "Social Anxiety"),
    ("specific_phobia", "Specific Phobia"),
]
METHOD_DISPLAY = {
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "dsm5": "\\textit{\\DSMV}",
    "dsm5+icd10": "\\textit{\\DualKB}",
    "icd10": "\\textit{\\ICDten}",
    "none": "\\textit{\\NoKB}",
    "random": "Random",
}
ROW_ORDER = ["ctgan","tvae","dsm5","dsm5+icd10","icd10","none","random"]
LLM_ORDER = ["icd10","dsm5","dsm5+icd10"]  # for delta table

def _fmt_ci(mu, lo, hi):
    if not (np.isfinite(mu) and np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"{mu:.3f} [{lo:.3f}, {hi:.3f}]"

def load_inputs():
    sum_csv = IN_DIR / "fidelity_summary.csv"
    raw_csv = IN_DIR / "fidelity_bootstrap_raw.csv"
    if not (sum_csv.exists() and raw_csv.exists()):
        raise FileNotFoundError(f"Missing inputs in {IN_DIR}. Expected fidelity_summary.csv and fidelity_bootstrap_raw.csv.")
    df_sum = pd.read_csv(sum_csv)
    df_raw = pd.read_csv(raw_csv)
    # normalize
    for c in ["disorder","method","metric"]:
        df_sum[c] = df_sum[c].str.lower()
        df_raw[c] = df_raw[c].str.lower()
    return df_sum, df_raw

def write_full_table(df_sum: pd.DataFrame):
    # Build LaTeX Table 1
    lines = []
    lines += [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Fidelity with 95\\% bootstrap confidence intervals. Best per metric within each disorder is \\textbf{bold}.}",
        "\\label{tab:fidelity_results_complete}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\rotatebox{70}{\\textbf{Disorder}} & "
        "\\rotatebox{70}{\\textbf{Method}} & "
        "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{JSD} $\\downarrow$}} & "
        "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{MAE\\_V} $\\downarrow$}} & "
        "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{ED$^2$} $\\downarrow$}} \\\\",
        "\\midrule",
    ]

    for slug, disp in DISP_ORDER:
        g = df_sum[df_sum["disorder"] == slug]
        if g.empty:
            continue
        # find best (lowest) per metric among available methods
        avail = [m for m in ROW_ORDER if (slug, m) in set(zip(g["disorder"], g["method"]))]
        def best_mask(metric):
            subset = g[g["metric"] == metric].set_index("method")
            vals = {m: subset.loc[m, "mean"] if m in subset.index else np.nan for m in avail}
            finite = [v for v in vals.values() if np.isfinite(v)]
            if not finite:
                return {m: False for m in avail}
            mn = np.nanmin(list(vals.values()))
            return {m: (np.isfinite(vals[m]) and abs(vals[m]-mn) <= 1e-12) for m in avail}

        bJ = best_mask("jsd")
        bM = best_mask("mae_v")
        bE = best_mask("ed2")

        first = True
        for m in avail:
            rowJ = g[(g["method"]==m) & (g["metric"]=="jsd")].iloc[0]
            rowM = g[(g["method"]==m) & (g["metric"]=="mae_v")].iloc[0]
            rowE = g[(g["method"]==m) & (g["metric"]=="ed2")].iloc[0]
            def cell(r, bold):
                s = _fmt_ci(r["mean"], r["ci_lo"], r["ci_hi"])
                return f"\\textbf{{{s}}}" if bold else s
            lead = f"\\multirow{{{len(avail)}}}{{*}}{{\\rotatebox{{90}}{{{disp}}}}}" if first else " "
            first = False
            lines.append(
                f"{lead} & {METHOD_DISPLAY.get(m,m)} & "
                f"{cell(rowJ, bJ[m])} & "
                f"{cell(rowM, bM[m])} & "
                f"{cell(rowE, bE[m])} \\\\"
            )
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}"]

    (OUT_DIR / "fidelity_results_complete.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT_DIR/'fidelity_results_complete.tex'}")

def write_delta_table(df_raw: pd.DataFrame):
    """
    Paired deltas vs No-KB across the SAME bootstrap index b, per disorder & metric.
    Output Table 2 LaTeX.
    """
    want_metrics = ["jsd","mae_v","ed2"]
    df = df_raw[df_raw["metric"].isin(want_metrics)].copy()
    # align on (disorder, metric, bootstrap)
    records = []
    for disorder in sorted(df["disorder"].unique()):
        for metric in want_metrics:
            base = df[(df["disorder"]==disorder) & (df["metric"]==metric) & (df["method"]=="none")]
            if base.empty:
                continue
            base = base.set_index("bootstrap")["value"]
            for meth in LLM_ORDER:
                cand = df[(df["disorder"]==disorder) & (df["metric"]==metric) & (df["method"]==meth)]
                if cand.empty:
                    continue
                cand = cand.set_index("bootstrap")["value"]
                n = int(min(len(base), len(cand)))
                if n < 2:
                    continue
                # truncate & pair
                a = cand.iloc[:n].to_numpy(float)
                b = base.iloc[:n].to_numpy(float)
                # Direction: "improvement" (higher-is-better deltas):
                # for distance metrics (JSD, ED2): delta = b - a
                # for MAE_V (a distance here): also lower is better -> b - a
                delta = b - a
                mu = float(np.mean(delta))
                lo, hi = np.quantile(delta, [0.025, 0.975])
                records.append({
                    "disorder": disorder, "method": meth, "metric": metric.upper(),
                    "mean": mu, "ci_lo": float(lo), "ci_hi": float(hi)
                })
    dfd = pd.DataFrame(records)

    # LaTeX
    title = "\\caption{Improvements over \\textit{None} with paired 95\\% CIs. Best per metric within each disorder is \\textbf{bold}.}"
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        title,
        "\\label{tab:delta_fidelity_narrow}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "\\rotatebox{70}{\\textbf{Disorder}} & "
        "\\rotatebox{70}{\\textbf{Method}} & "
        "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{$\\Delta$JSD} $\\uparrow$}} & "
        "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{$\\Delta$MAE$_V$} $\\uparrow$}} & "
        "\\rotatebox{70}{\\parbox[b]{1.35cm}{\\centering \\textbf{$\\Delta$ED$^2$} $\\uparrow$}} \\\\",
        "\\midrule",
    ]
    for slug, disp in DISP_ORDER:
        g = dfd[dfd["disorder"]==slug]
        if g.empty:
            continue
        # best per metric (highest) among ICD-10, DSM-V, Dual-KB
        def best_mask(metric):
            gm = g[g["metric"]==metric]
            if gm.empty:
                return {m: False for m in LLM_ORDER}
            sub = gm.set_index("method")["mean"].to_dict()
            if not sub:
                return {m: False for m in LLM_ORDER}
            mx = np.nanmax(list(sub.values()))
            return {m: (m in sub and np.isfinite(sub[m]) and abs(sub[m]-mx)<=1e-12) for m in LLM_ORDER}

        bJ = best_mask("JSD")
        bM = best_mask("MAE_V")
        bE = best_mask("ED2")

        first = True
        for m in LLM_ORDER:
            rowJ = g[(g["method"]==m) & (g["metric"]=="JSD")].iloc[0]
            rowM = g[(g["method"]==m) & (g["metric"]=="MAE_V")].iloc[0]
            rowE = g[(g["method"]==m) & (g["metric"]=="ED2")].iloc[0]
            def cell(r, bold):
                s = _fmt_ci(r["mean"], r["ci_lo"], r["ci_hi"])
                return f"\\textbf{{{s}}}" if bold else s
            lead = f"\\multirow{{3}}{{*}}{{\\rotatebox{{90}}{{{disp}}}}}" if first else " "
            first = False
            lines.append(
                f"{lead} & {METHOD_DISPLAY.get(m,m)} & "
                f"{cell(rowJ, bJ[m])} & "
                f"{cell(rowM, bM[m])} & "
                f"{cell(rowE, bE[m])} \\\\"
            )
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}"]

    (OUT_DIR / "delta_fidelity_narrow.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT_DIR/'delta_fidelity_narrow.tex'}")

def main():
    df_sum, df_raw = load_inputs()
    write_full_table(df_sum)
    write_delta_table(df_raw)

if __name__ == "__main__":
    main()
