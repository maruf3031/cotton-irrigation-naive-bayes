"""
utils.py — একাধিক স্ক্রিপ্টে যে কাজগুলো বারবার লাগে, সেগুলো এখানে একবার লেখা।

কেন আলাদা ফাইল?
  ধাপ ২-এর প্রতিটা স্ক্রিপ্টে (CV, ablation, noise) একই ভাবে ডেটা লোড, outlier বাদ,
  আর একই মডেলগুলো লাগবে। প্রতিটায় আলাদা করে লিখলে কোথাও একটু ভিন্ন হয়ে যেতে পারে,
  আর তাহলে তুলনা ন্যায্য হয় না। এক জায়গায় লিখলে সবাই হুবহু একই জিনিস ব্যবহার করে।
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# ডেটা লোড
# ---------------------------------------------------------------------------
def load_d1():
    """
    D1 লোড করে X (moisture, temp) আর y (pump) ফেরত দেয়।
    crop কলাম বাদ — সব row-এ "cotton", কোনো তথ্য দেয় না (রিভিউয়ার A-৫)।
    """
    df = pd.read_csv(config.D1_FILE, encoding="utf-8-sig")
    X = df[["moisture", "temp"]].astype(float)
    y = df["pump"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Outlier বাদ — শুধু training অংশ থেকে
# ---------------------------------------------------------------------------
def trim_outliers_train(X_train, y_train, q=config.OUTLIER_QUANTILE):
    """
    পুরনো পদ্ধতিতে পুরো ডেটা দেখে outlier বাদ দেওয়া হতো (data leakage)।
    এখন সীমা ঠিক হয় শুধু training অংশ দেখে, আর শুধু training থেকেই বাদ যায়।
    Test অংশ কখনো ছোঁয়া হয় না — বাস্তবে নতুন সেন্সর রিডিং তো আমরা বাদ দিতে পারি না।
    """
    keep = np.ones(len(X_train), dtype=bool)
    for col in ["moisture", "temp"]:
        keep &= (X_train[col] < X_train[col].quantile(q)).to_numpy()
    return X_train[keep], y_train[keep]


# ---------------------------------------------------------------------------
# Feature তৈরি: raw + log (পেপারের মতো)
# ---------------------------------------------------------------------------
def add_log_features(X):
    """moisture, temp-এর পাশে log1p(moisture), log1p(temp) যোগ করে (পুরনো C2 configuration)।"""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, np.log1p(X)])


def log_features(X):
    """
    শুধু log1p(moisture), log1p(temp) — চূড়ান্ত বাছাই (ablation-এর C4)।
    কারণ: accuracy প্রায় একই (0.984 বনাম 0.985), কিন্তু raw+log-এর redundancy নেই
    (রিভিউয়ার A-৪)। মডেলও সহজ — মাত্র ২টা feature।
    """
    X = np.asarray(X, dtype=float)
    return np.log1p(X)


# ---------------------------------------------------------------------------
# Threshold rule — "moisture একটা সীমার বেশি হলে পাম্প ON"
# ---------------------------------------------------------------------------
class MoistureThresholdRule(BaseEstimator, ClassifierMixin):
    """
    সবচেয়ে সাধারণ নিয়ম-ভিত্তিক সিস্টেম (রিভিউয়ার A-৩ যেটা চেয়েছেন)।
    সীমাটা training ডেটা থেকে শেখা হয়: OFF-এর সর্বোচ্চ moisture আর ON-এর সর্বনিম্ন
    moisture-এর মাঝামাঝি। এভাবে test ডেটা দেখে সীমা বসানো হয় না।
    """

    def __init__(self, fixed=None):
        # fixed=None → সীমা ডেটা থেকে শেখা হবে; fixed=450 → কৃষক হাতে 450 বসিয়েছেন
        self.fixed = fixed

    def fit(self, X, y):
        m = np.asarray(X, dtype=float)[:, 0]   # প্রথম কলাম = moisture
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        if self.fixed is not None:
            self.threshold_ = float(self.fixed)
        else:
            off_max = m[y == 0].max()
            on_min = m[y == 1].min()
            self.threshold_ = (off_max + on_min) / 2
        return self

    def predict(self, X):
        m = np.asarray(X, dtype=float)[:, 0]
        return (m >= self.threshold_).astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------------------
# চূড়ান্ত Naive Bayes মডেল (log feature → GaussianNB)
# ---------------------------------------------------------------------------
def make_nb():
    return make_pipeline(FunctionTransformer(log_features),
                         GaussianNB(var_smoothing=config.GNB_VAR_SMOOTHING))


# ---------------------------------------------------------------------------
# তুলনার সব মডেল
# ---------------------------------------------------------------------------
def make_models():
    """
    প্রতিটা মডেল একটা pipeline: আগে feature তৈরি/scaling, তারপর মডেল।
    Pipeline ব্যবহারের সুবিধা: scaling-এর গড়/SD শুধু training অংশ থেকে শেখে (leakage নেই)।
    LR, KNN, SVM দূরত্ব বা ওজন দিয়ে কাজ করে, তাই scaling লাগে; গাছ-ভিত্তিক মডেল আর NB-তে লাগে না।
    """
    seed = config.RANDOM_SEED
    return {
        "Threshold rule":      MoistureThresholdRule(),
        "Gaussian NB":         make_nb(),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "Decision Tree":       DecisionTreeClassifier(random_state=seed),
        "KNN (k=5)":           make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF)":           make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed)),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=seed),
    }
