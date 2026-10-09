"""
ধাপ ২খ — Ablation study (Naive Bayes, D1)

"Ablation" মানে: মডেলের একেকটা অংশ সরিয়ে দেখা, ফলাফলে কী বদলায়।
এতে বোঝা যায় কোন ধাপটা আসলে কাজে লাগছে আর কোনটা অপ্রয়োজনীয়।

রিভিউয়ারের যে মন্তব্যের উত্তর:
  A-4: "raw আর log দুটো feature একসাথে রাখলে redundancy হয় এবং Naive Bayes-এর
        conditional independence assumption ভাঙে। Outlier removal আর feature
        পরিবর্তনের প্রভাব ablation দিয়ে দেখাও।"
  A-5: crop কলামের কোনো প্রভাব আছে কিনা।

পরীক্ষার নিয়ম:
  সব configuration একই 100টা train/test ভাগে চালানো হয় (একই random seed),
  তাই পার্থক্য শুধু configuration-এর কারণে — ভাগের ভাগ্যের কারণে নয়।

চালানো:  python src\\03_ablation.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import utils   # noqa: E402

warnings.filterwarnings("ignore")
config.ensure_dirs()

X, y = utils.load_d1()


# ---------------------------------------------------------------------------
# Feature তৈরির উপায়গুলো
# ---------------------------------------------------------------------------
def make_features(Xdf, kind):
    """kind অনুযায়ী কোন feature যাবে তা ঠিক করে।"""
    m, t = Xdf["moisture"].to_numpy(), Xdf["temp"].to_numpy()
    lm, lt = np.log1p(m), np.log1p(t)
    if kind == "moisture only":
        return np.column_stack([m])
    if kind == "raw":
        return np.column_stack([m, t])
    if kind == "log":
        return np.column_stack([lm, lt])
    if kind == "raw + log":
        return np.column_stack([m, t, lm, lt])
    if kind == "raw + log + crop":          # পেপারের হুবহু feature set (crop_cotton সবসময় 1)
        return np.column_stack([m, t, np.ones_like(m), lm, lt])
    raise ValueError(kind)


# (নাম, feature-এর ধরন, outlier বাদ দেবে কিনা)
CONFIGS = [
    ("C1: পেপারের মতো (raw+log+crop, outlier বাদ)", "raw + log + crop", True),
    ("C2: crop বাদ (raw+log, outlier বাদ)",          "raw + log",        True),
    ("C3: শুধু raw (outlier বাদ)",                    "raw",              True),
    ("C4: শুধু log (outlier বাদ)",                    "log",              True),
    ("C5: শুধু raw (outlier বাদ নয়)",                 "raw",              False),
    ("C6: শুধু log (outlier বাদ নয়)",                 "log",              False),
    ("C7: raw+log (outlier বাদ নয়)",                  "raw + log",        False),
    ("C8: শুধু moisture (temp বাদ)",                  "moisture only",    True),
]

cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS,
                             random_state=config.RANDOM_SEED)

rows = []
for fold_id, (tr, te) in enumerate(cv.split(X, y)):
    X_tr_full, y_tr_full = X.iloc[tr], y.iloc[tr]
    X_tr_trim, y_tr_trim = utils.trim_outliers_train(X_tr_full, y_tr_full)
    X_te, y_te = X.iloc[te], y.iloc[te]

    for name, kind, trim in CONFIGS:
        X_tr, y_tr = (X_tr_trim, y_tr_trim) if trim else (X_tr_full, y_tr_full)
        model = GaussianNB(var_smoothing=config.GNB_VAR_SMOOTHING)
        model.fit(make_features(X_tr, kind), y_tr)
        pred = model.predict(make_features(X_te, kind))
        rows.append({"fold": fold_id, "config": name,
                     "accuracy": accuracy_score(y_te, pred),
                     "f1_macro": f1_score(y_te, pred, average="macro")})

scores = pd.DataFrame(rows)
summary = (scores.groupby("config", sort=False)
                 .agg(accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                      f1_macro_mean=("f1_macro", "mean"), f1_macro_sd=("f1_macro", "std"))
                 .reset_index())
summary.to_csv(config.RESULTS_TABLES / "03_ablation_summary.csv", index=False, float_format="%.4f")
scores.to_csv(config.RESULTS_TABLES / "03_ablation_fold_scores.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------------------
# Redundancy কতটা? raw আর log feature-এর correlation
# ---------------------------------------------------------------------------
r_m = np.corrcoef(X["moisture"], np.log1p(X["moisture"]))[0, 1]
r_t = np.corrcoef(X["temp"], np.log1p(X["temp"]))[0, 1]

# ---------------------------------------------------------------------------
# দেখানো
# ---------------------------------------------------------------------------
print("=" * 72)
print("Ablation ফলাফল — Gaussian NB, 10×10-fold CV (গড় ± SD)")
print("=" * 72)
print(f"{'Configuration':<46}{'Accuracy':>13}{'F1 (macro)':>13}")
for _, r in summary.iterrows():
    print(f"{r.config:<46}{r.accuracy_mean:>7.3f} ± {r.accuracy_sd:.3f}"
          f"{r.f1_macro_mean:>7.3f} ± {r.f1_macro_sd:.3f}")

print()
print("Redundancy (raw আর log-এর correlation; 1-এর কাছে মানে প্রায় একই তথ্য):")
print(f"  moisture ↔ log(moisture): r = {r_m:.3f}")
print(f"  temp     ↔ log(temp)    : r = {r_t:.3f}")

c1 = summary.iloc[0].accuracy_mean
c2 = summary.iloc[1].accuracy_mean
print()
print("কীভাবে পড়বেন:")
print(f"  • C1 বনাম C2 (crop আছে/নেই): {c1:.3f} বনাম {c2:.3f} → crop-এর প্রভাব",
      "নেই" if abs(c1 - c2) < 1e-9 else "আছে")
print("  • C3 বনাম C4 বনাম C2: raw, log, নাকি দুটো একসাথে — কোনটা ভালো")
print("  • C3 বনাম C5 (বা C4 বনাম C6): outlier বাদ দেওয়ায় লাভ হয় কিনা")
print("  • C8: temperature বাদ দিলে কী হয়")
print()
print("✅ সেভ হয়েছে: results\\tables\\03_ablation_summary.csv")
