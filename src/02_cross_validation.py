"""
ধাপ ২ক — Repeated stratified 10-fold cross-validation + মডেল তুলনা (D1)

রিভিউয়ারের যে মন্তব্যের উত্তর:
  A-2: "80:20 split আর ছোট test set নির্ভরযোগ্য নয় — k-fold CV করে mean ± SD দাও"
  A-3 / B: "শুধু Naive Bayes কেন? threshold rule, LR, DT, KNN-এর সাথে তুলনা দাও"
  A-5: "Crop কলাম অর্থহীন" → বাদ দেওয়া হয়েছে (utils.load_d1)

পুরনো পদ্ধতি থেকে যা বদলেছে:
  1. একবার 80:20 ভাগের বদলে: ডেটাকে 10 ভাগে ভাগ, প্রতিবার 1 ভাগে test, বাকি 9-এ train
     → 10টা ফলাফল। পুরো প্রক্রিয়া 10 বার ভিন্ন এলোমেলো ভাগে → মোট 100টা ফলাফল।
  2. "Stratified": প্রতিটা ভাগে ON/OFF-এর অনুপাত মূল ডেটার মতো (75% ON) রাখা হয়।
  3. Outlier বাদ এখন প্রতিবার শুধু training অংশ থেকে (leakage নেই)।
  4. crop_cotton কলাম বাদ।

চালানো:  python src\\02_cross_validation.py   (প্রায় ১–২ মিনিট লাগতে পারে)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import utils   # noqa: E402

warnings.filterwarnings("ignore")
config.ensure_dirs()

X, y = utils.load_d1()
models = utils.make_models()
cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS,
                             random_state=config.RANDOM_SEED)

print(f"ডেটা: {len(X)} row, feature = {list(X.columns)}")
print(f"পদ্ধতি: {config.CV_REPEATS} × {config.CV_FOLDS}-fold = "
      f"{config.CV_REPEATS * config.CV_FOLDS}টা train/test ভাগ, মডেল {len(models)}টা\n")

# ---------------------------------------------------------------------------
# প্রতিটা ভাগে প্রতিটা মডেল train + test
# ---------------------------------------------------------------------------
rows = []
for fold_id, (tr, te) in enumerate(cv.split(X, y)):
    X_tr, y_tr = utils.trim_outliers_train(X.iloc[tr], y.iloc[tr])
    X_te, y_te = X.iloc[te], y.iloc[te]

    for name, model in models.items():
        m = clone(model)              # প্রতিবার একদম নতুন (অশিক্ষিত) মডেল
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        prob = m.predict_proba(X_te)[:, 1]
        rows.append({
            "fold": fold_id,
            "model": name,
            "accuracy": accuracy_score(y_te, pred),
            "precision_ON": precision_score(y_te, pred, zero_division=0),
            "recall_ON": recall_score(y_te, pred, zero_division=0),
            "f1_ON": f1_score(y_te, pred, zero_division=0),
            "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y_te, prob),
        })
    if (fold_id + 1) % 20 == 0:
        print(f"  {fold_id + 1}/{config.CV_FOLDS * config.CV_REPEATS} ভাগ শেষ…")

scores = pd.DataFrame(rows)
scores.to_csv(config.RESULTS_TABLES / "02_cv_fold_scores.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------------------
# সারসংক্ষেপ: mean ± SD, আর 100টা ফলাফলের মাঝের 95% (2.5–97.5 percentile)
# ---------------------------------------------------------------------------
metrics = ["accuracy", "precision_ON", "recall_ON", "f1_ON", "f1_macro", "roc_auc"]
summary_rows = []
for name in models:
    s = scores[scores["model"] == name]
    row = {"model": name}
    for met in metrics:
        v = s[met].to_numpy()
        row[f"{met}_mean"] = v.mean()
        row[f"{met}_sd"] = v.std(ddof=1)
        row[f"{met}_p2.5"] = np.percentile(v, 2.5)
        row[f"{met}_p97.5"] = np.percentile(v, 97.5)
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows).sort_values("accuracy_mean", ascending=False)
summary.to_csv(config.RESULTS_TABLES / "02_cv_summary.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------------------
# স্ক্রিনে সুন্দর করে দেখানো
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("ফলাফল (100টা ভাগের গড় ± SD)   [বন্ধনীতে: 100টা ফলাফলের মাঝের 95%]")
print("=" * 78)
print(f"{'মডেল':<22}{'Accuracy':>26}{'F1 (macro)':>16}{'ROC-AUC':>14}")
for _, r in summary.iterrows():
    acc = f"{r.accuracy_mean:.3f} ± {r.accuracy_sd:.3f} [{r['accuracy_p2.5']:.2f}–{r['accuracy_p97.5']:.2f}]"
    print(f"{r.model:<22}{acc:>26}{r.f1_macro_mean:>10.3f} ± {r.f1_macro_sd:.3f}{r.roc_auc_mean:>9.3f}")

nb = summary[summary.model == "Gaussian NB"].iloc[0]
rule = summary[summary.model == "Threshold rule"].iloc[0]
print()
print("মূল পর্যবেক্ষণ:")
print(f"  • Gaussian NB    : {nb.accuracy_mean:.3f} ± {nb.accuracy_sd:.3f}")
print(f"  • Threshold rule : {rule.accuracy_mean:.3f} ± {rule.accuracy_sd:.3f}")
print("    (একটা মাত্র moisture সীমা দিয়ে — এটাই এই ডেটাসেটের মূল সত্য)")
print()
print("✅ সেভ হয়েছে: results\\tables\\02_cv_summary.csv (সারসংক্ষেপ)")
print("             results\\tables\\02_cv_fold_scores.csv (100টা ভাগের সব ফলাফল)")
