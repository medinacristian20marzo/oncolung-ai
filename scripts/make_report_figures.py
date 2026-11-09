#!/usr/bin/env python3
"""Figures and summary numbers for the report.

Usage: python scripts/make_report_figures.py [package_root] [output_dir]
Refits M0 and M1, repeats the 200-replicate paired bootstrap (seed 42) and
redraws the mutation-expression comparison from the matched table.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy.stats import chi2, mannwhitneyu, false_discovery_control

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "report"
OUT.mkdir(parents=True, exist_ok=True)

GENES = ["EGFR", "KRAS", "TP53", "STK11"]
TARGETS = ["OS_MONTHS", "OS_STATUS"]
CLINICAL = ["age_10", "sex_M", "stage_II", "stage_III", "stage_IV"]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
STAGE_COLORS = {"I": "#86b6ef", "II": "#3987e5", "III": "#1c5cab", "IV": "#0d366b"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "axes.linewidth": 0.6, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

df = pd.read_csv(ROOT / "tcga_luad_day0.csv")
X = pd.DataFrame({
    "OS_MONTHS": df.OS_MONTHS, "OS_STATUS": df.OS_STATUS, "age_10": df.age / 10,
    "sex_M": (df.sex == "M").astype(float),
    **{f"stage_{s}": (df.stage == s).astype(float) for s in ["II", "III", "IV"]},
    **{g: df[g].astype(float) for g in GENES},
})


def fit(data):
    model = CoxPHFitter(baseline_estimation_method="breslow", penalizer=0.0)
    return model.fit(data, "OS_MONTHS", "OS_STATUS")


m0, m1 = fit(X[TARGETS + CLINICAL]), fit(X[TARGETS + CLINICAL + GENES])

# ---- paired bootstrap optimism ------------------------------------------
rng = np.random.RandomState(42)
optimism = []
for _ in range(200):
    idx = rng.choice(X.index, size=len(X), replace=True)
    boot = X.loc[idx].reset_index(drop=True)
    b0, b1 = fit(boot[TARGETS + CLINICAL]), fit(boot[TARGETS + CLINICAL + GENES])
    o0 = b0.score(X[TARGETS + CLINICAL], scoring_method="concordance_index")
    o1 = b1.score(X[TARGETS + CLINICAL + GENES], scoring_method="concordance_index")
    optimism.append((b0.concordance_index_ - o0, b1.concordance_index_ - o1,
                     (b1.concordance_index_ - b0.concordance_index_) - (o1 - o0)))
optimism = np.array(optimism)

# ---- Figure 1: Kaplan-Meier ---------------------------------------------
TICKS = [0, 24, 48, 72, 96, 120]


def km_panel(ax, column, groups, labels, colors, title, show_ci):
    rows = []
    for g, lab, col in zip(groups, labels, colors):
        sub = df[df[column] == g]
        km = KaplanMeierFitter().fit(sub.OS_MONTHS, sub.OS_STATUS)
        km.plot_survival_function(ax=ax, color=col, lw=1.6, ci_show=show_ci, ci_alpha=0.12,
                                  show_censors=True, label="_nolegend_",
                                  censor_styles={"marker": "|", "ms": 3.5, "mew": 0.6})
        ax.plot([], [], color=col, lw=1.6, label=f"{lab} (n = {len(sub)})")
        rows.append((lab, col, [int((sub.OS_MONTHS >= t).sum()) for t in TICKS]))
    ax.set(xlim=(0, 126), ylim=(0, 1.02), xticks=TICKS, xlabel="Months from start of follow-up",
           ylabel="Overall survival probability", title=title)
    ax.yaxis.grid(True, color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
    handles = [h for h in ax.get_lines() if not h.get_label().startswith("_") and len(h.get_xdata()) == 0]
    ax.legend(handles=handles, frameon=False, loc="upper right", handlelength=1.4)
    y = -0.30
    ax.text(-0.09, y, "No. at risk", transform=ax.transAxes, ha="right", va="center",
            fontsize=7, color=MUTED)
    for lab, col, counts in rows:
        y -= 0.075
        ax.text(-0.09, y, lab, transform=ax.transAxes, ha="right", va="center", fontsize=7, color=INK)
        for t, c in zip(TICKS, counts):
            ax.text(t, y, str(c), transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=7, color=INK)


fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9))
km_panel(axes[0], "stage", ["I", "II", "III", "IV"], [f"Stage {s}" for s in ["I", "II", "III", "IV"]],
         [STAGE_COLORS[s] for s in ["I", "II", "III", "IV"]], "a   Pathological stage", False)
km_panel(axes[1], "KRAS", [0, 1], ["KRAS wild-type", "KRAS mutated"], [BLUE, ORANGE], "b   KRAS status", True)
fig.subplots_adjust(left=0.16, right=0.99, top=0.93, bottom=0.36, wspace=0.52)
fig.savefig(OUT / "fig1_km.pdf"); fig.savefig(OUT / "fig1_km.png"); plt.close(fig)

# ---- Figure 2: hazard ratios and optimism --------------------------------
LABELS = {"age_10": "Age (per 10 years)", "sex_M": "Male sex", "stage_II": "Stage II vs I",
          "stage_III": "Stage III vs I", "stage_IV": "Stage IV vs I", "EGFR": "EGFR mutated",
          "KRAS": "KRAS mutated", "TP53": "TP53 mutated", "STK11": "STK11 mutated"}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw={"width_ratios": [1.45, 1]})
order = CLINICAL + GENES
for i, name in enumerate(order):
    y = len(order) - 1 - i
    for model, col, dy, mk in [(m0, BLUE, 0.17, "o"), (m1, ORANGE, -0.17, "s")]:
        if name not in model.summary.index:
            continue
        r = model.summary.loc[name]
        ax.plot([r["exp(coef) lower 95%"], r["exp(coef) upper 95%"]], [y + dy] * 2, color=col, lw=1.2)
        ax.plot(r["exp(coef)"], y + dy, mk, color=col, ms=4.5, mec="white", mew=0.6)
ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
ax.axhline(3.5, color="#e6e5e1", lw=0.6)
ax.set_xscale("log"); ax.set_xticks([0.75, 1, 1.5, 2, 3, 4, 6])
ax.set_xticklabels(["0.75", "1", "1.5", "2", "3", "4", "6"]); ax.minorticks_off()
ax.set_yticks(range(len(order))); ax.set_yticklabels([LABELS[n] for n in order][::-1], color=INK)
ax.tick_params(axis="y", length=0)
ax.set(xlabel="Hazard ratio (95% CI), log scale", title="a   Adjusted hazard ratios", ylim=(-0.6, 8.6))
ax.plot([], [], "o-", color=BLUE, ms=4.5, lw=1.2, label="M0, clinical")
ax.plot([], [], "s-", color=ORANGE, ms=4.5, lw=1.2, label="M1, clinical + mutations")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

d = optimism[:, 2]
bx.hist(d, bins=22, color=BLUE, edgecolor="white", lw=0.6)
bx.axvline(0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
bx.axvline(d.mean(), color=INK, lw=1.1)
bx.text(d.mean() + 0.002, bx.get_ylim()[1] * 0.97, f"mean {d.mean():+.4f}", fontsize=7.5, color=INK, va="top")
bx.set(xlabel="Optimism in ΔC per bootstrap replicate", ylabel="Replicates",
       title="b   Optimism in ΔC")
fig.subplots_adjust(left=0.165, right=0.99, top=0.92, bottom=0.25, wspace=0.26)
fig.savefig(OUT / "fig2_models.pdf"); fig.savefig(OUT / "fig2_models.png"); plt.close(fig)

# ---- Figure 3: expression by mutation status -----------------------------
rna = pd.read_csv(ROOT / "outputs" / "rna" / "clinical_mutation_rna_293.csv")
pvals, stats = [], {}
for g in GENES:
    v = np.log2(rna[f"{g}_RSEM"] + 1)
    x1, x0 = v[rna[g] == 1].to_numpy(), v[rna[g] == 0].to_numpy()
    u, p = mannwhitneyu(x1, x0, alternative="two-sided", method="asymptotic", use_continuity=True)
    pvals.append(p)
    stats[g] = {"n0": len(x0), "n1": len(x1), "median0": float(np.median(x0)), "median1": float(np.median(x1)),
                "iqr0": np.quantile(x0, [.25, .75]).tolist(), "iqr1": np.quantile(x1, [.25, .75]).tolist(),
                "diff": float(np.median(x1) - np.median(x0)), "rank_biserial": float(2 * u / (len(x1) * len(x0)) - 1),
                "p": float(p), "x0": x0, "x1": x1}
for g, q in zip(GENES, false_discovery_control(np.array(pvals), method="bh")):
    stats[g]["q"] = float(q)


def fmt_q(q):
    if q >= 0.001:
        return f"q = {q:.3f}"
    m, e = f"{q:.1e}".split("e")
    return f"q = {m} × 10$^{{{int(e)}}}$"


jit = np.random.RandomState(42)
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.75))
for ax, g in zip(axes, GENES):
    s = stats[g]
    for pos, vals, col in [(0, s["x0"], BLUE), (1, s["x1"], ORANGE)]:
        ax.scatter(pos + jit.uniform(-0.17, 0.17, len(vals)), vals, s=5, color=col, alpha=0.35, lw=0)
        ax.boxplot(vals, positions=[pos], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=INK, lw=0.8),
                   medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=INK, lw=0.8),
                   capprops=dict(color=INK, lw=0.8))
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"WT\nn = {s['n0']}", f"Mut\nn = {s['n1']}"], color=INK)
    ax.set_title(g, loc="left"); ax.text(1.0, 1.035, fmt_q(s["q"]), transform=ax.transAxes,
                                          ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.yaxis.grid(True, color="#e6e5e1", lw=0.5); ax.set_axisbelow(True); ax.set_xlim(-0.55, 1.55)
axes[0].set_ylabel("log$_2$(RSEM + 1)")
fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.17, wspace=0.30)
fig.savefig(OUT / "fig3_expression.pdf"); fig.savefig(OUT / "fig3_expression.png"); plt.close(fig)

# ---- numbers quoted in the report ----------------------------------------
lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
kras = df.KRAS == 1
summary = {
    "n": len(df), "deaths": int(df.OS_STATUS.sum()),
    "median_follow_up_reverse_km": KaplanMeierFitter().fit(df.OS_MONTHS, 1 - df.OS_STATUS).median_survival_time_,
    "median_os": KaplanMeierFitter().fit(df.OS_MONTHS, df.OS_STATUS).median_survival_time_,
    "logrank_stage_p": multivariate_logrank_test(df.OS_MONTHS, df.stage, df.OS_STATUS).p_value,
    "logrank_kras_p": logrank_test(df.OS_MONTHS[kras], df.OS_MONTHS[~kras], df.OS_STATUS[kras], df.OS_STATUS[~kras]).p_value,
    "c_apparent": [m0.concordance_index_, m1.concordance_index_],
    "optimism_mean": optimism.mean(0).tolist(),
    "c_corrected": [m0.concordance_index_ - optimism[:, 0].mean(), m1.concordance_index_ - optimism[:, 1].mean()],
    "delta_c_corrected": (m1.concordance_index_ - m0.concordance_index_) - optimism[:, 2].mean(),
    "share_replicates_delta_optimism_positive": float((optimism[:, 2] > 0).mean()),
    "lrt": {"statistic": lr, "df": 4, "p": float(chi2.sf(lr, 4))},
    "aic_partial": [m0.AIC_partial_, m1.AIC_partial_],
    "hazard_ratios_m1": m1.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(4).to_dict("index"),
    "hazard_ratios_m0": m0.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(4).to_dict("index"),
    "expression": {g: {k: v for k, v in s.items() if k not in ("x0", "x1")} for g, s in stats.items()},
}
(OUT / "report_numbers.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
