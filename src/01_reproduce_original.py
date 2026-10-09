"""
ধাপ ১ — পেপারের পুরনো ফলাফল (97%) হুবহু আবার বের করা

কেন এই ধাপ?
  নতুন কিছু করার আগে নিশ্চিত হতে হবে যে পেপারের সংখ্যাগুলো সত্যিই এই কোড থেকে আসে।
  এটাকে বলে "reproduction"। পরের সব তুলনা এই ফলাফলকে শুরুর বিন্দু ধরে হবে।

এই স্ক্রিপ্ট পুরনো D:\\Cotton\\NB.py-এর ধাপগুলো হুবহু অনুসরণ করে,
শুধু path আর গোছানো বদলানো হয়েছে। ছবি বানানো এখানে বাদ (ধাপ ৪-এ হবে)।

চালানো:  python src\\01_reproduce_original.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

config.ensure_dirs()

# ---------------------------------------------------------------------------
# ১. ডেটা লোড
# ---------------------------------------------------------------------------
# encoding="utf-8-sig": ফাইলের শুরুতে একটা লুকানো চিহ্ন (BOM) আছে, এটা সেটা সরিয়ে দেয়।
df = pd.read_csv(config.D1_FILE, encoding="utf-8-sig")
print(f"মূল ডেটা: {len(df)} row, কলাম = {list(df.columns)}")

# ---------------------------------------------------------------------------
# ২. Outlier বাদ (পেপারের মতো: moisture আর temp-এর উপরের 1%)
# ---------------------------------------------------------------------------
# ⚠️ পুরনো পদ্ধতির একটা দুর্বলতা: এখানে পুরো ডেটা দেখে সীমা ঠিক করা হচ্ছে,
#    train/test ভাগ করার আগেই। একে বলে data leakage। ধাপ ২-এ এটা ঠিক করা হবে।
#    এখানে শুধু পুরনো ফলাফল মেলানোর জন্য হুবহু রাখা হলো।
q = config.OUTLIER_QUANTILE
df = df[df["moisture"] < df["moisture"].quantile(q)]
df = df[df["temp"] < df["temp"].quantile(q)]
print(f"Outlier বাদের পর: {len(df)} row")

# ---------------------------------------------------------------------------
# ৩. One-hot encoding (crop) আর log transform
# ---------------------------------------------------------------------------
# ⚠️ crop-এর মান সবসময় "cotton", তাই crop_cotton কলাম সবসময় 1 — কোনো তথ্য দেয় না
#    (রিভিউয়ার A-র ৫ নম্বর মন্তব্য)। ধাপ ২-এ বাদ দেওয়া হবে।
df = pd.get_dummies(df, columns=["crop"], dtype=int)
df["log_moisture"] = np.log1p(df["moisture"])   # log1p(x) = log(1 + x)
df["log_temp"] = np.log1p(df["temp"])

# ---------------------------------------------------------------------------
# ৪. Feature (X) আর target (y) আলাদা, তারপর 80:20 ভাগ
# ---------------------------------------------------------------------------
X = df.drop(columns=["pump"])
y = df["pump"]
print(f"Feature-গুলো: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
)
print(f"Train: {len(X_train)} row, Test: {len(X_test)} row")

# ---------------------------------------------------------------------------
# ৫. Gaussian Naive Bayes — train আর test
# ---------------------------------------------------------------------------
model = GaussianNB(var_smoothing=config.GNB_VAR_SMOOTHING)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print()
print("=" * 50)
print("ফলাফল")
print("=" * 50)
print(f"Train accuracy: {acc_train:.4f}")
print(f"Test accuracy : {acc_test:.4f}")
print()
print("Confusion matrix (সারি = আসল, কলাম = অনুমান; 0 = OFF, 1 = ON):")
print(pd.DataFrame(cm, index=["আসল OFF", "আসল ON"], columns=["অনুমান OFF", "অনুমান ON"]))
print()
print(classification_report(y_test, y_test_pred, digits=2))

# ---------------------------------------------------------------------------
# ৬. পেপারের সংখ্যার সাথে মেলানো
# ---------------------------------------------------------------------------
# পেপারের Table II–IV থেকে: test-এ 39টা sample, accuracy 0.97, confusion matrix [[11,1],[0,27]]
paper_cm = np.array([[11, 1], [0, 27]])
checks = {
    "Test sample সংখ্যা = 39": len(y_test) == 39,
    "Test accuracy = 0.97 (দুই দশমিকে)": round(acc_test, 2) == 0.97,
    "Confusion matrix = [[11,1],[0,27]]": np.array_equal(cm, paper_cm),
}
print("=" * 50)
print("পেপারের সাথে মিল আছে কি?")
print("=" * 50)
for name, ok in checks.items():
    print(f"  [{'মিলেছে' if ok else 'মেলেনি'}] {name}")

# ---------------------------------------------------------------------------
# ৭. ফলাফল সেভ
# ---------------------------------------------------------------------------
report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T
report.to_csv(config.RESULTS_TABLES / "01_original_classification_report.csv", float_format="%.4f")
pd.DataFrame(cm, index=["actual_OFF", "actual_ON"], columns=["pred_OFF", "pred_ON"]).to_csv(
    config.RESULTS_TABLES / "01_original_confusion_matrix.csv"
)

print()
if all(checks.values()):
    print("✅ পেপারের ফলাফল হুবহু পুনরুৎপাদন হয়েছে।")
else:
    print("⚠️  কিছু সংখ্যা পেপারের সাথে মেলেনি — উপরে দেখুন।")
print("   টেবিল সেভ হয়েছে: results\\tables\\01_original_*.csv")
