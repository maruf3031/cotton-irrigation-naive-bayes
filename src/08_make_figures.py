"""
ধাপ ৪ — পেপারের সব ছবি (300 dpi PNG) এক স্ক্রিপ্টে

রিভিউয়ারের যে মন্তব্যের উত্তর:
  A (minor-3): "Figure caption আরও পূর্ণাঙ্গ করো, পরীক্ষার অবস্থা স্পষ্ট করো"
  → প্রতিটা ছবির জন্য একটা প্রস্তাবিত caption results\\figures\\figure_captions.txt-এ লেখা হয়।
  পুরনো পেপারে Fig. 1 (flow diagram) উল্লেখ ছিল কিন্তু ছবিটা ছিল না — এখানে বানানো হলো।

রঙের নিয়ম (সব ছবিতে একই, যাতে পাঠক একবার শিখে নেন):
  নীল = Gaussian NB (আমাদের মডেল), কমলা = Threshold rule, সবুজাভ = Decision Tree,
  ধূসর = বাকি মডেল। রঙ ছাড়াও নাম সরাসরি লেখা থাকে (রঙকানা পাঠক ও সাদা-কালো ছাপার জন্য)।

আগে চালাতে হবে: 02, 03, 05, 07 (এদের CSV থেকে ছবি বানানো হয়)
চালানো:  python src\\08_make_figures.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # স্ক্রিনে না দেখিয়ে সরাসরি ফাইলে সেভ
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import utils   # noqa: E402

warnings.filterwarnings("ignore")
config.ensure_dirs()
T, F = config.RESULTS_TABLES, config.RESULTS_FIGURES

# ---------------------------------------------------------------------------
# সাধারণ চেহারা
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHT_GRAY, INK, INK2 = "#9a9993", "#d6d5d0", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def color_of(model):
    if model.startswith("Gaussian NB"):
        return BLUE
    if "rule" in model.lower():
        return ORANGE
    if model.startswith("Decision Tree"):
        return AQUA
    return GRAY


captions = {}


def save(fig, name, caption):
    fig.savefig(F / name)
    plt.close(fig)
    captions[name] = caption
    print(f"  ✓ {name}")


# ---------------------------------------------------------------------------
# Fig 1 — Workflow diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
ax.grid(False)


def box(x, y, w, h, text, edge=INK2, fill="#f7f7f5", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.0,
            color=INK, weight="bold" if bold else "normal", linespacing=1.3)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))


# সারি ১: D1
box(0.1, 3.3, 1.9, 1.2, "D1: AutoIrrigation\n(Kaggle, simulated)\n200 records\nmoisture, temp → pump", bold=False)
box(2.4, 3.3, 1.9, 1.2, "Preprocessing\n• drop constant 'crop'\n• log1p transform\n• train-only outlier trim")
box(4.7, 3.3, 2.3, 1.2, "Models (7)\nThreshold rule, Gaussian NB,\nLR, DT, KNN, SVM, RF")
box(7.4, 3.3, 2.5, 1.2, "Evaluation\n• 10×10 stratified CV\n• ablation (8 configs)\n• sensor-noise test\n• IoT cost benchmark")
for x1, x2 in [(2.0, 2.4), (4.3, 4.7), (7.0, 7.4)]:
    arrow(x1, 3.9, x2, 3.9)
# সারি ২: D4
box(0.1, 0.9, 1.9, 1.2, "D4: Bushland, TX\n(USDA-ARS lysimeters)\n8 seasons, 19 lysimeters\nreal cotton field")
box(2.4, 0.9, 1.9, 1.2, "Feature building\n(only day t−1 data)\nwater deficit, ET,\nstorage change, weather")
box(4.7, 0.9, 2.3, 1.2, "Same models\n+ deficit-threshold rule\n+ 'always no' baseline")
box(7.4, 0.9, 2.5, 1.2, "External validation\n• leave-one-year-out\n• balanced acc., F1,\n  ROC-AUC")
for x1, x2 in [(2.0, 2.4), (4.3, 4.7), (7.0, 7.4)]:
    arrow(x1, 1.5, x2, 1.5)
box(7.4, 2.3, 2.5, 0.65, "Irrigation decision:\npump ON / OFF", edge=BLUE, fill="#eef4fc", bold=True)
arrow(8.65, 3.3, 8.65, 2.95); arrow(8.65, 2.1, 8.65, 2.3)
ax.text(0.1, 4.75, "A  Primary dataset", fontsize=8.5, weight="bold", color=INK)
ax.text(0.1, 2.35, "B  External real-field dataset", fontsize=8.5, weight="bold", color=INK)
save(fig, "fig01_workflow.png",
     "Workflow of the study. (A) The primary simulated dataset (D1) is preprocessed and seven "
     "classifiers are compared with repeated stratified 10-fold cross-validation, an ablation study, "
     "a sensor-noise robustness test and a computational-cost benchmark. (B) The same models are "
     "externally validated on real cotton-field lysimeter data from Bushland, Texas (D4) using "
     "leave-one-year-out evaluation; all D4 features use only information available up to the previous day.")

# ---------------------------------------------------------------------------
# Fig 2 — D1: moisture বনাম temp (threshold দৃশ্যমান)
# ---------------------------------------------------------------------------
X, y = utils.load_d1()
rule = utils.MoistureThresholdRule().fit(X, y)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for cls, col, mk, lab in [(0, ORANGE, "o", "Pump OFF (n = 50)"), (1, BLUE, "^", "Pump ON (n = 150)")]:
    s = X[y == cls]
    ax.scatter(s.moisture, s.temp, s=22, c=col, marker=mk, edgecolors="white", linewidths=0.6,
               label=lab, zorder=3)
ax.axvline(rule.threshold_, color=INK2, lw=1, ls="--", zorder=2)
ax.text(rule.threshold_ + 12, 6.5, f"learned threshold ≈ {rule.threshold_:.0f}", fontsize=7.5, color=INK2)
ax.set_xlabel("Soil-moisture sensor reading (0–1023; higher = drier)")
ax.set_ylabel("Air temperature (°C)")
ax.set_ylim(5, 49)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
save(fig, "fig02_d1_scatter.png",
     "Distribution of the 200 records of the AutoIrrigation dataset (D1). The pump state is completely "
     f"separated by a single soil-moisture value (pump OFF ≤ 493, pump ON ≥ 503; dashed line at ≈ {rule.threshold_:.0f}), "
     "whereas air temperature does not differ between classes. Higher sensor readings correspond to drier soil.")

# ---------------------------------------------------------------------------
# Fig 3 — log feature-এর histogram ও Q-Q plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
for col_i, (feat, lab) in enumerate([("moisture", "log(1 + moisture)"), ("temp", "log(1 + temperature)")]):
    v = np.log1p(X[feat])
    ax = axes[0, col_i]
    ax.hist(v, bins=20, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.set_xlabel(lab); ax.set_ylabel("Count")
    ax.set_title(f"({'ab'[col_i]}) Histogram of {lab}", loc="left")
    ax = axes[1, col_i]
    (osm, osr), (slope, inter, r) = stats.probplot(v, dist="norm")
    ax.plot(osm, osr, "o", ms=3, color=BLUE, mec="white", mew=0.4)
    ax.plot(osm, slope * osm + inter, color=INK2, lw=1)
    ax.set_xlabel("Theoretical normal quantiles"); ax.set_ylabel("Ordered values")
    ax.set_title(f"({'cd'[col_i]}) Normal Q-Q plot, r = {r:.3f}", loc="left")
fig.tight_layout()
save(fig, "fig03_log_distributions.png",
     "Distributions of the log-transformed predictors used by the final Gaussian Naive Bayes model "
     "(all 200 records of D1). (a, b) Histograms; (c, d) normal Q-Q plots with the probability-plot "
     "correlation coefficient r. Log-moisture is bimodal because the two pump classes occupy separate "
     "moisture ranges, so the Gaussian assumption holds within classes rather than marginally.")

# ---------------------------------------------------------------------------
# Fig 4 — চূড়ান্ত NB-র confusion matrix (10-fold CV, সব ভাগ মিলিয়ে)
# ---------------------------------------------------------------------------
cm = np.zeros((2, 2), dtype=int)
for tr, te in StratifiedKFold(10, shuffle=True, random_state=config.RANDOM_SEED).split(X, y):
    Xtr, ytr = utils.trim_outliers_train(X.iloc[tr], y.iloc[tr])
    pred = utils.make_nb().fit(Xtr, ytr).predict(X.iloc[te])
    cm += confusion_matrix(y.iloc[te], pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.grid(False)
ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f2f7fd", "#1c5cab"]))
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > cm.max() / 2 else INK)
ax.set_xticks([0, 1], ["OFF", "ON"]); ax.set_yticks([0, 1], ["OFF", "ON"])
ax.set_xlabel("Predicted pump state"); ax.set_ylabel("Actual pump state")
for s in ax.spines.values():
    s.set_visible(False)
acc = np.trace(cm) / cm.sum()
save(fig, "fig04_confusion_matrix_nb.png",
     f"Confusion matrix of the final Gaussian Naive Bayes model (log-transformed moisture and temperature) "
     f"on D1, pooled over the ten test folds of one stratified 10-fold cross-validation (n = {cm.sum()}; "
     f"accuracy = {acc:.3f}). Outlier trimming was fitted on training folds only.")
pd.DataFrame(cm, index=["actual_OFF", "actual_ON"], columns=["pred_OFF", "pred_ON"]).to_csv(
    T / "08_confusion_matrix_nb_cv.csv")

# ---------------------------------------------------------------------------
# Fig 5 — D1 মডেল তুলনা (dot + error bar)
# ---------------------------------------------------------------------------
s = pd.read_csv(T / "02_cv_summary.csv").sort_values("accuracy_mean")
fig, ax = plt.subplots(figsize=(5.2, 3.0))
yy = np.arange(len(s))
for i, r in enumerate(s.itertuples()):
    c = color_of(r.model)
    ax.errorbar(r.accuracy_mean, i, xerr=r.accuracy_sd, fmt="o", ms=7, color=c, ecolor=c,
                elinewidth=1.4, capsize=0, mec="white", mew=0.8, zorder=3)
    ax.text(1.06, i, f"{r.accuracy_mean:.3f} ± {r.accuracy_sd:.3f}", va="center", fontsize=7.5,
            color=INK2, transform=ax.get_yaxis_transform())
ax.set_yticks(yy, s.model)
for lbl in ax.get_yticklabels():
    if lbl.get_text().startswith("Gaussian NB"):
        lbl.set_weight("bold")
ax.set_xlabel("Accuracy (mean ± SD over 100 test folds)")
ax.set_xlim(0.9, 1.01)
ax.grid(axis="y", visible=False)
save(fig, "fig05_d1_model_comparison.png",
     "Accuracy of seven classifiers on D1 under 10-times-repeated stratified 10-fold cross-validation "
     "(100 test folds; points = mean, bars = ±1 SD). Outlier trimming and feature scaling were fitted on "
     "training folds only. The single-threshold rule, decision tree and random forest separate D1 perfectly, "
     "reflecting the rule-based origin of its labels.")

# ---------------------------------------------------------------------------
# Fig 6 — Ablation
# ---------------------------------------------------------------------------
a = pd.read_csv(T / "03_ablation_summary.csv")
eng = {
    "C1": "C1 raw + log + crop, trimmed (original)", "C2": "C2 raw + log, trimmed",
    "C3": "C3 raw, trimmed", "C4": "C4 log, trimmed (final)", "C5": "C5 raw, untrimmed",
    "C6": "C6 log, untrimmed", "C7": "C7 raw + log, untrimmed", "C8": "C8 moisture only, trimmed",
}
a["label"] = a.config.str[:2].map(eng)
a = a.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(5.4, 3.2))
for i, r in a.iterrows():
    final = r.label.startswith("C4")
    c = BLUE if final else GRAY
    ax.errorbar(r.accuracy_mean, i, xerr=r.accuracy_sd, fmt="o", ms=7 if final else 6, color=c,
                ecolor=c, elinewidth=1.3, mec="white", mew=0.8, zorder=3)
ax.set_yticks(range(len(a)), a.label)
for lbl in ax.get_yticklabels():
    if lbl.get_text().startswith("C4"):
        lbl.set_weight("bold")
ax.set_xlabel("Accuracy of Gaussian NB (mean ± SD, 100 test folds)")
ax.set_xlim(0.93, 1.01)
ax.grid(axis="y", visible=False)
save(fig, "fig06_ablation.png",
     "Ablation of the Gaussian Naive Bayes pipeline on D1 (identical 100 cross-validation splits for all "
     "configurations). Removing the constant 'crop' variable has no effect (C1 vs C2); raw and log versions "
     "of the same variable are highly correlated (r = 0.91 for moisture, 0.98 for temperature), and using only "
     "the log features (C4, selected) performs within 0.1 percentage points of the redundant raw + log set.")

# ---------------------------------------------------------------------------
# Fig 7 — Sensor noise
# ---------------------------------------------------------------------------
n = pd.read_csv(T / "05_noise_scores.csv")
m = n.groupby(["model", "noise_pct"]).accuracy.mean().unstack()
fig, ax = plt.subplots(figsize=(5.6, 3.4))
highlight = ["Gaussian NB", "Threshold rule", "Decision Tree", "Rule (fixed 550)"]
for model in m.index:
    if model not in highlight:
        ax.plot(m.columns, m.loc[model], color=LIGHT_GRAY, lw=1.2, zorder=1)
for model in highlight:
    c = color_of(model)
    ls = "--" if "fixed" in model else "-"
    ax.plot(m.columns, m.loc[model], color=c, lw=2, ls=ls, marker="o", ms=5, mec="white", mew=0.7, zorder=3)
# নামগুলো লাইনের শেষে, একটার ওপর আরেকটা যেন না পড়ে (ন্যূনতম ফাঁক রেখে সাজানো)
ends = sorted(((m.loc[k].iloc[-1], k) for k in highlight), reverse=True)
gap, placed = 0.0065, []
for val, k in ends:
    pos = val if not placed else min(val, placed[-1] - gap)
    placed.append(pos)
    ax.text(10.3, pos, k, va="center", fontsize=7.5, color=INK)
ax.set_xticks(m.columns, [f"{c}%" for c in m.columns])
ax.set_xlabel("Sensor noise added to soil moisture (SD as % of 0–1023 range)")
ax.set_ylabel("Accuracy (mean, 100 test folds)")
ax.set_xlim(-0.5, 10.2)
save(fig, "fig07_noise_robustness.png",
     "Robustness to simulated soil-moisture sensor noise on D1 (Gaussian noise added to training and test "
     "readings, labels unchanged; 20×5-fold cross-validation per level). The learned-threshold rule and the "
     "decision tree, which are perfect on clean data, degrade fastest; Gaussian NB, logistic regression and SVM "
     "decline gradually and end within 0.01 of each other at 10% noise. A manually set "
     "threshold that is slightly wrong (550 instead of ≈500, dashed) is worse than all learned models even "
     "without noise. Grey lines: logistic regression, KNN, SVM, random forest and the 450 threshold.")

# ---------------------------------------------------------------------------
# Fig 8 — Bushland (আসল মাঠ): দুটো প্যানেল, একই অক্ষে দুটো মাপ নয়
# ---------------------------------------------------------------------------
b = pd.read_csv(T / "07_bushland_summary.csv")
b = b[b.model != "Always 'no irrigation'"].sort_values("roc_auc_mean").reset_index(drop=True)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, met, title, lo in [(axes[0], "balanced_accuracy", "(a) Balanced accuracy", 0.45),
                           (axes[1], "roc_auc", "(b) ROC-AUC", 0.45)]:
    for i, r in b.iterrows():
        c = color_of(r.model)
        ax.errorbar(r[f"{met}_mean"], i, xerr=r[f"{met}_std"], fmt="o", ms=6.5, color=c, ecolor=c,
                    elinewidth=1.3, mec="white", mew=0.8, zorder=3)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.text(0.505, -0.95, "chance", fontsize=7, color=INK2)
    ax.set_ylim(-1.2, len(b) - 0.5)
    ax.set_title(title, loc="left"); ax.set_xlim(lo, 0.85)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(b)), b.model)
for lbl in axes[0].get_yticklabels():
    if lbl.get_text().startswith("Gaussian NB"):
        lbl.set_weight("bold")
fig.supxlabel("Mean ± SD over 8 held-out seasons (leave-one-year-out)", fontsize=8.5)
fig.tight_layout()
save(fig, "fig08_bushland_validation.png",
     "External validation on real cotton-field data (USDA-ARS Bushland, Texas; 19 lysimeter-seasons, "
     "2,584 in-season days, 12.4% irrigation days). Each season-year was held out in turn. All models are "
     "only modestly better than chance (dotted line), showing that real irrigation decisions are much harder "
     "to predict than the simulated D1 labels; Gaussian NB performs on par with heavier models and better "
     "than the deficit-threshold rule. A model that never irrigates reaches 87% accuracy but 0.50 balanced "
     "accuracy (not shown).")

# ---------------------------------------------------------------------------
# Caption ফাইল
# ---------------------------------------------------------------------------
with open(F / "figure_captions.txt", "w", encoding="utf-8") as f:
    for name, cap in captions.items():
        f.write(f"{name}\n{cap}\n\n")
print()
print("✅ ৮টা ছবি সেভ হয়েছে: results\\figures\\")
print("   প্রস্তাবিত caption: results\\figures\\figure_captions.txt")
