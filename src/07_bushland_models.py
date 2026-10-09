"""
ধাপ ৩খ — আসল cotton মাঠে (Bushland, Texas) মডেলগুলোর পরীক্ষা

প্রশ্ন: "আজ সেচ দেওয়া হবে কি?" — শুধু গতকাল পর্যন্ত জানা তথ্য দিয়ে।
Label: Irrigation Token (গবেষকরা আসলেই সেদিন সেচ দিয়েছিলেন কিনা)।

রিভিউয়ারের যে মন্তব্যের উত্তর:
  A-1 / B: "ছোট, simulated ডেটা — আসল মাঠের ডেটায় যাচাই দরকার"
  A-2: শক্ত evaluation (এখানে leave-one-year-out)
  A-3: threshold rule সহ অন্য মডেলের সাথে তুলনা

গুরুত্বপূর্ণ সতর্কতা (data leakage এড়াতে):
  • সেচের দিনের "Gauge Precip"-এ sprinkler-এর পানিও ধরা পড়ে (যেমন 2008 NE, দিন 152:
    সেচ = 1, precip = 11 mm)। তাই আজকের কোনো মাপ feature-এ নেই — সব feature গতকাল
    বা তার আগের দিনের।
  • 2000_SE lysimeter-এ Irrigation Token সারা বছর 0, অথচ README বলছে সেখানে সেচ দেওয়া হয়েছে
    → token লেখা হয়নি। তাই এই lysimeter বাদ।
  • শুধু মৌসুমের মূল সময় (DOY 130–265, মে-এর মাঝামাঝি থেকে সেপ্টেম্বর) — এর আগের
    বপনপূর্ব সেচ ভিন্ন ধরনের সিদ্ধান্ত।

Feature (সবগুলো গতকাল পর্যন্ত):
  deficit_mm        — শেষবার ভালো পানি পড়ার (বৃষ্টি/সেচ ≥ 5 mm) পর থেকে মোট ET (mm)
                      → মাটি থেকে কতটা পানি খরচ হয়েছে। সেচ-সূচির সবচেয়ে প্রচলিত ভিত্তি।
  days_since_water  — শেষবার ভালো পানি পড়ার পর কত দিন
  et_3d_mm          — গত ৩ দিনের গড় ET
  storage_change_3d — গত ৩ দিনে মাটির পানি কতটা বাড়ল/কমল (mm)
  air_temp_c, rh_pct, wind_ms — গতকালের আবহাওয়া
  doy               — মৌসুমের কোন সময় (গাছের বয়সের প্রতিনিধি)

পরীক্ষার পদ্ধতি — Leave-one-year-out:
  ৮টা বছরের প্রতিটাকে একবার করে test বানানো হয়, বাকি ৭ বছরে train।
  এটা কঠিন কিন্তু সৎ পরীক্ষা: মডেল কি একদম নতুন একটা মৌসুমে কাজ করে?

চালানো:  python src\\07_bushland_models.py   (প্রায় ১ মিনিট)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score)

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

warnings.filterwarnings("ignore")
config.ensure_dirs()

SEASON = (130, 265)
WET_MM = 5.0          # এর বেশি পানি পড়লে মাটি "ভিজেছে" ধরা হয়
FEATURES = ["deficit_mm", "days_since_water", "et_3d_mm", "storage_change_3d",
            "air_temp_c", "rh_pct", "wind_ms", "doy"]

# ---------------------------------------------------------------------------
# ১. ডেটা লোড ও feature তৈরি
# ---------------------------------------------------------------------------
raw = pd.read_csv(config.DATA_PROCESSED / "d4_bushland_daily.csv")
raw = raw[raw.lysimeter != "2000_SE"]           # token লেখা হয়নি (উপরে ব্যাখ্যা)


def build_features(g):
    """একটা lysimeter-মৌসুমের পুরো বছরের দিনগুলো থেকে feature (সব গতকাল পর্যন্ত)।"""
    g = g.sort_values("doy").copy()
    et = g["et_mm"].clip(0, 15)                  # maintenance-এর দিনের অবাস্তব ET (−1184!) কাটা
    water = g["precip_mm"].fillna(0)             # বৃষ্টি + sprinkler সেচ (gauge-এ ধরা পড়ে)
    wet = water >= WET_MM

    deficit, days = [], []
    d_acc, n_acc = 0.0, 0
    for e, w in zip(et.fillna(0), wet):
        # এই দিনের শেষে অবস্থা হালনাগাদ
        if w:
            d_acc, n_acc = 0.0, 0
        else:
            d_acc += e
            n_acc += 1
        deficit.append(d_acc)
        days.append(n_acc)
    g["deficit_end"] = deficit
    g["days_end"] = days

    out = pd.DataFrame({
        "lysimeter": g["lysimeter"], "year": g["year"], "doy": g["doy"],
        "irrigation": g["irrigation"],
        # shift(1) = গতকালের মান → আজকের সিদ্ধান্তে আজকের মাপ ঢোকে না
        "deficit_mm": g["deficit_end"].shift(1),
        "days_since_water": g["days_end"].shift(1),
        "et_3d_mm": et.shift(1).rolling(3, min_periods=1).mean(),
        "storage_change_3d": g["storage_mm"].shift(1) - g["storage_mm"].shift(4),
        "air_temp_c": g["air_temp_c"].shift(1),
        "rh_pct": g["rh_pct"].shift(1),
        "wind_ms": g["wind_ms"].shift(1),
    })
    return out


data = pd.concat([build_features(g) for _, g in raw.groupby("lysimeter")], ignore_index=True)
data = data[(data.doy >= SEASON[0]) & (data.doy <= SEASON[1])].reset_index(drop=True)
data.to_csv(config.DATA_PROCESSED / "d4_bushland_features.csv", index=False, float_format="%.4f")

X, y, groups = data[FEATURES], data["irrigation"].astype(int), data["year"]
print(f"ডেটা: {len(data)} দিন, {data.lysimeter.nunique()}টা lysimeter-মৌসুম, {groups.nunique()}টা বছর")
print(f"সেচের দিন: {y.sum()} ({y.mean() * 100:.1f}%) — বাকিগুলো সেচ ছাড়া")
print(f"Missing মান (গড়ে): {X.isna().mean().mean() * 100:.1f}% → train অংশের median দিয়ে পূরণ\n")


# ---------------------------------------------------------------------------
# ২. মডেল
# ---------------------------------------------------------------------------
class DeficitThresholdRule(BaseEstimator, ClassifierMixin):
    """
    প্রচলিত নিয়ম: "জমে থাকা ঘাটতি (deficit) একটা সীমা ছাড়ালে সেচ দাও"।
    সীমাটা training ডেটা থেকে বেছে নেওয়া হয় — যে সীমায় F1 সবচেয়ে বেশি।
    """

    def fit(self, X, y):
        d = np.asarray(X)[:, 0]
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        candidates = np.unique(np.round(d[~np.isnan(d)], 1))
        scores = [f1_score(y, (d >= t).astype(int), zero_division=0) for t in candidates]
        self.threshold_ = candidates[int(np.argmax(scores))]
        return self

    def predict(self, X):
        return (np.nan_to_num(np.asarray(X)[:, 0]) >= self.threshold_).astype(int)

    def predict_proba(self, X):
        d = np.nan_to_num(np.asarray(X)[:, 0])
        p = 1 / (1 + np.exp(-(d - self.threshold_)))      # সীমা থেকে দূরত্বকে 0–1-এ আনা (AUC-এর জন্য)
        return np.column_stack([1 - p, p])


class AlwaysNo(BaseEstimator, ClassifierMixin):
    """সবচেয়ে বোকা মডেল: কখনো সেচ দেয় না। Accuracy কতটা বিভ্রান্তিকর তা দেখাতে।"""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


seed = config.RANDOM_SEED
imp = SimpleImputer(strategy="median")
models = {
    "Always 'no irrigation'": AlwaysNo(),
    "Deficit threshold rule": make_pipeline(clone(imp), DeficitThresholdRule()),
    # priors=[0.5, 0.5]: অন্য মডেলের class_weight="balanced"-এর সমতুল্য — সেচের দিন কম (12%) বলে
    # মডেল যেন "কখনো সেচ নয়"-এর দিকে ঝুঁকে না পড়ে। ন্যায্য তুলনার জন্য সবাইকে একই সুবিধা।
    "Gaussian NB":            make_pipeline(clone(imp), GaussianNB(priors=[0.5, 0.5])),
    "Logistic Regression":    make_pipeline(clone(imp), StandardScaler(),
                                            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    "Decision Tree":          make_pipeline(clone(imp), DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed)),
    "KNN (k=5)":              make_pipeline(clone(imp), StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM (RBF)":              make_pipeline(clone(imp), StandardScaler(),
                                            SVC(probability=True, class_weight="balanced", random_state=seed)),
    "Random Forest":          make_pipeline(clone(imp), RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                                                class_weight="balanced", random_state=seed)),
}

# ---------------------------------------------------------------------------
# ৩. Leave-one-year-out
# ---------------------------------------------------------------------------
rows = []
for tr, te in LeaveOneGroupOut().split(X, y, groups):
    year = int(groups.iloc[te].iloc[0])
    for name, model in models.items():
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        prob = m.predict_proba(X.iloc[te])[:, 1]
        yt = y.iloc[te]
        rows.append({
            "test_year": year, "model": name,
            "accuracy": accuracy_score(yt, pred),
            "balanced_accuracy": balanced_accuracy_score(yt, pred),
            "precision": precision_score(yt, pred, zero_division=0),
            "recall": recall_score(yt, pred, zero_division=0),
            "f1": f1_score(yt, pred, zero_division=0),
            "roc_auc": roc_auc_score(yt, prob),
            "pr_auc": average_precision_score(yt, prob),
        })
    print(f"  test বছর {year} শেষ")

scores = pd.DataFrame(rows)
scores.to_csv(config.RESULTS_TABLES / "07_bushland_by_year.csv", index=False, float_format="%.4f")

metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
summary = scores.groupby("model", sort=False)[metrics].agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary = summary.reset_index()
summary.to_csv(config.RESULTS_TABLES / "07_bushland_summary.csv", index=False, float_format="%.4f")

# ---------------------------------------------------------------------------
# ৪. দেখানো
# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("আসল cotton মাঠ (Bushland) — leave-one-year-out, ৮টা বছরের গড় ± SD")
print("=" * 96)
print(f"{'মডেল':<25}{'Accuracy':>14}{'Bal. Acc.':>14}{'Recall':>14}{'F1':>14}{'ROC-AUC':>14}")
for _, r in summary.iterrows():
    print(f"{r.model:<25}"
          + "".join(f"{r[f'{m}_mean']:>8.3f}±{r[f'{m}_std']:.2f}" for m in
                    ["accuracy", "balanced_accuracy", "recall", "f1", "roc_auc"]))

print()
print("কীভাবে পড়বেন:")
print("  • মাত্র ~12% দিনে সেচ হয়, তাই 'কখনো সেচ নয়' মডেলও ~88% accuracy পায় —")
print("    এজন্য এখানে Accuracy দিয়ে বিচার করা যাবে না। মূল মাপ: Balanced accuracy, F1, ROC-AUC।")
print("  • Recall = আসল সেচের দিনগুলোর কত ভাগ ধরতে পেরেছে (কম হলে গাছ পানি-কষ্টে পড়বে)।")
print("  • ROC-AUC: 0.5 = এলোমেলো অনুমান, 1.0 = নিখুঁত।")
print()
print("✅ সেভ হয়েছে: results\\tables\\07_bushland_summary.csv, 07_bushland_by_year.csv")
print("             data\\processed\\d4_bushland_features.csv")
