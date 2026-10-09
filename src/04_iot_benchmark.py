"""
ধাপ ২গ — IoT benchmark: মডেলগুলো কতটা হালকা আর দ্রুত?

রিভিউয়ারের যে মন্তব্যের উত্তর:
  A-6: "পেপারে দাবি করা হয়েছে মডেলটা real-time, low-power IoT-র জন্য উপযুক্ত,
        কিন্তু inference time বা memory footprint-এর কোনো সংখ্যা নেই।"

কী মাপা হয় (প্রতিটা মডেলের জন্য):
  1. Training time        — পুরো D1-এ একবার train করতে কত সময়
  2. Inference time       — একটা মাত্র সেন্সর রিডিং থেকে সিদ্ধান্ত দিতে কত মাইক্রোসেকেন্ড (µs)
                            (IoT-তে রিডিং আসে একটা একটা করে, তাই এটাই আসল মাপ)
  3. Model size           — সেভ করা মডেল ফাইল কত বাইট
  4. সংরক্ষিত সংখ্যা       — মডেল চালাতে কতগুলো সংখ্যা মনে রাখতে হয়
                            (মাইক্রোকন্ট্রোলারে লিখতে গেলে এটাই আসল মেমোরি খরচ)

⚠️ সততার কথা: এই মাপ আপনার PC-তে (Python-এ)। মাইক্রোকন্ট্রোলারে (ESP32/Arduino)
   আসল সময় ভিন্ন হবে। তাই পেপারে লিখতে হবে "PC-তে আপেক্ষিক তুলনা", আর সংরক্ষিত
   সংখ্যার হিসাব দিয়ে দেখাতে হবে কোন মডেল ছোট ডিভাইসে বসানো সম্ভব।

চালানো:  python src\\04_iot_benchmark.py   (প্রায় ১ মিনিট)
"""

import io
import sys
import time
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import utils   # noqa: E402

warnings.filterwarnings("ignore")
config.ensure_dirs()

X, y = utils.load_d1()
X_tr, y_tr = utils.trim_outliers_train(X, y)
models = utils.make_models()

N_SINGLE = 2000          # একটা একটা করে কতবার অনুমান করে সময় মাপা হবে
rng = np.random.default_rng(config.RANDOM_SEED)
sample_idx = rng.integers(0, len(X), N_SINGLE)
X_np = X.to_numpy()


def count_stored_numbers(model):
    """মডেল চালাতে কতগুলো সংখ্যা মনে রাখতে হয় (মোটামুটি হিসাব)।"""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    name = type(est).__name__
    if name == "MoistureThresholdRule":
        return 1                                                   # শুধু সীমাটা
    if name == "GaussianNB":
        return est.theta_.size + est.var_.size + est.class_prior_.size   # গড়, variance, prior
    if name == "LogisticRegression":
        n = est.coef_.size + est.intercept_.size
        return n + 2 * est.coef_.shape[1]                          # + scaler-এর গড় ও SD
    if name == "DecisionTreeClassifier":
        return est.tree_.node_count * 2                            # প্রতি node-এ সীমা + feature/ফল
    if name == "RandomForestClassifier":
        return sum(t.tree_.node_count * 2 for t in est.estimators_)
    if name == "KNeighborsClassifier":
        return est._fit_X.size + len(est._y) + 2 * est._fit_X.shape[1]   # পুরো training ডেটা!
    if name == "SVC":
        return est.support_vectors_.size + est.dual_coef_.size + est.intercept_.size + 2 * est.support_vectors_.shape[1]
    return np.nan


rows = []
for name, model in models.items():
    m = clone(model)

    # ১. Training time (৫ বার মেপে মাঝের মান)
    fit_times = []
    for _ in range(5):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - t0)

    # ২. একটা রিডিং থেকে অনুমানের সময়
    _ = m.predict(X.iloc[[0]])                     # প্রথমবারের "গরম হওয়া" বাদ
    single_times = []
    for i in sample_idx:
        row = X.iloc[[i]]
        t0 = time.perf_counter()
        m.predict(row)
        single_times.append(time.perf_counter() - t0)

    # ৩. মডেল ফাইলের আকার (মেমোরিতে সেভ করে মাপা)
    buf = io.BytesIO()
    joblib.dump(m, buf)
    size_bytes = buf.getbuffer().nbytes

    rows.append({
        "model": name,
        "train_time_ms": np.median(fit_times) * 1000,
        "inference_us_median": np.median(single_times) * 1e6,
        "inference_us_p95": np.percentile(single_times, 95) * 1e6,
        "model_size_kb": size_bytes / 1024,
        "stored_numbers": count_stored_numbers(m),
    })
    print(f"  {name} মাপা শেষ")

res = pd.DataFrame(rows)
res.to_csv(config.RESULTS_TABLES / "04_iot_benchmark.csv", index=False, float_format="%.3f")

# চূড়ান্ত NB মডেল সেভ (পরে ছবি/ব্যাখ্যার কাজে লাগবে)
nb = utils.make_nb().fit(X_tr, y_tr)
joblib.dump(nb, config.RESULTS_MODELS / "final_gaussian_nb.pkl")
gnb = nb.steps[-1][1]

print()
print("=" * 80)
print(f"IoT benchmark — মেশিন: {platform.processor() or platform.machine()}, Python {platform.python_version()}")
print("=" * 80)
print(f"{'মডেল':<22}{'Train (ms)':>11}{'অনুমান µs':>12}{'(95%)':>9}{'ফাইল KB':>10}{'সংরক্ষিত সংখ্যা':>17}")
for _, r in res.sort_values("stored_numbers").iterrows():
    print(f"{r.model:<22}{r.train_time_ms:>11.2f}{r.inference_us_median:>12.1f}"
          f"{r.inference_us_p95:>9.1f}{r.model_size_kb:>10.2f}{int(r.stored_numbers):>17,}")

print()
print("চূড়ান্ত Naive Bayes মডেলে যা যা সংরক্ষিত আছে (এগুলোই পুরো মডেল!):")
print(f"  Class prior (OFF, ON)        : {np.round(gnb.class_prior_, 4).tolist()}")
print(f"  গড় log(moisture), log(temp)  : OFF = {np.round(gnb.theta_[0], 4).tolist()}, "
      f"ON = {np.round(gnb.theta_[1], 4).tolist()}")
print(f"  Variance                     : OFF = {np.round(gnb.var_[0], 4).tolist()}, "
      f"ON = {np.round(gnb.var_[1], 4).tolist()}")
print("  → মোট 10টা সংখ্যা। একটা সিদ্ধান্তের জন্য লাগে ~২টা log আর ~২০টা গুণ/যোগ —")
print("    যেকোনো ছোট মাইক্রোকন্ট্রোলার (যেমন ESP32, Arduino) সহজে করতে পারে।")
print()
print("✅ সেভ হয়েছে: results\\tables\\04_iot_benchmark.csv")
print("             results\\models\\final_gaussian_nb.pkl")
