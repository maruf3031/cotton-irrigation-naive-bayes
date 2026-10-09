"""
ধাপ ২ঘ — Sensor noise আর ভুল threshold: আসল মাঠের মতো অবস্থায় কে কতটা টেকে?

কেন এই পরীক্ষা?
  D1-এ moisture-এর মান একদম নিখুঁত, তাই "moisture ≥ ~500 হলে ON" নিয়মেই 100% আসে।
  কিন্তু আসল সেন্সর রিডিং কাঁপে (noise)। আর আসল কৃষক জানেন না সঠিক সীমা 500 —
  কেউ 450 বসান, কেউ 550। এই পরীক্ষা দুটো প্রশ্নের উত্তর দেয়:
    1. রিডিং-এ noise থাকলে কোন মডেল কতটা ভুল করে?
    2. হাতে বসানো সীমা একটু ভুল হলে নিয়ম কতটা খারাপ করে? (ML মডেল সীমা নিজে শেখে)

পদ্ধতি:
  • Noise = moisture-এ এলোমেলো (Gaussian) কাঁপুনি। মাত্রা: সেন্সরের পুরো রেঞ্জের
    (0–1023) 0%, 2%, 5%, 10% — অর্থাৎ SD ≈ 0, ±20, ±51, ±102 ইউনিট।
  • Noise train আর test দুই অংশেই যোগ হয় (বাস্তবে সব রিডিংই সেন্সর থেকে আসে),
    কিন্তু label (pump) বদলায় না — মাটির আসল অবস্থা একই, শুধু রিডিং ভুল।
  • প্রতিটা noise মাত্রায় 20 × 5-fold = 100টা train/test ভাগ।

রিভিউয়ারের যে মন্তব্যের উত্তর: A-1 (বাস্তব অবস্থায় প্রযোজ্যতা), A-3 (threshold baseline)

চালানো:  python src\\05_noise_robustness.py   (প্রায় ২–৪ মিনিট)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import utils   # noqa: E402

warnings.filterwarnings("ignore")
config.ensure_dirs()

X, y = utils.load_d1()
SENSOR_RANGE = 1023.0     # 10-bit সেন্সর রিডিং: 0–1023

models = utils.make_models()
# হাতে বসানো (ভুল হতে পারে এমন) সীমার নিয়ম — তুলনার জন্য
models["Rule (fixed 450)"] = utils.MoistureThresholdRule(fixed=450)
models["Rule (fixed 550)"] = utils.MoistureThresholdRule(fixed=550)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=config.RANDOM_SEED)
splits = list(cv.split(X, y))
rng = np.random.default_rng(config.RANDOM_SEED)


def add_noise(Xdf, level):
    """moisture কলামে Gaussian noise যোগ করে; মান 0–1023-এর মধ্যে রাখে।"""
    Xn = Xdf.copy()
    if level > 0:
        noise = rng.normal(0, level * SENSOR_RANGE, len(Xn))
        Xn["moisture"] = np.clip(Xn["moisture"] + noise, 0, SENSOR_RANGE)
    return Xn


rows = []
for level in config.NOISE_LEVELS:
    for split_id, (tr, te) in enumerate(splits):
        X_tr = add_noise(X.iloc[tr], level)
        X_te = add_noise(X.iloc[te], level)
        X_tr, y_tr = utils.trim_outliers_train(X_tr, y.iloc[tr])
        y_te = y.iloc[te]
        for name, model in models.items():
            m = clone(model).fit(X_tr, y_tr)
            pred = m.predict(X_te)
            rows.append({"noise_pct": int(level * 100), "split": split_id, "model": name,
                         "accuracy": accuracy_score(y_te, pred),
                         "f1_macro": f1_score(y_te, pred, average="macro")})
    print(f"  noise {int(level * 100)}% শেষ")

scores = pd.DataFrame(rows)
scores.to_csv(config.RESULTS_TABLES / "05_noise_scores.csv", index=False, float_format="%.4f")

mean = scores.pivot_table(index="model", columns="noise_pct", values="accuracy", aggfunc="mean")
sd = scores.pivot_table(index="model", columns="noise_pct", values="accuracy", aggfunc="std")
order = list(models.keys())
mean, sd = mean.loc[order], sd.loc[order]

summary = mean.round(4).astype(str) + " ± " + sd.round(4).astype(str)
summary.columns = [f"noise_{c}%" for c in summary.columns]
summary.to_csv(config.RESULTS_TABLES / "05_noise_summary.csv")

print()
print("=" * 78)
print("Accuracy (গড় ± SD, 100টা ভাগ) — moisture-এ sensor noise-এর মাত্রা অনুযায়ী")
print("=" * 78)
header = f"{'মডেল':<22}" + "".join(f"{f'noise {c}%':>14}" for c in mean.columns)
print(header)
for name in order:
    line = f"{name:<22}"
    for c in mean.columns:
        line += f"{mean.loc[name, c]:>8.3f}±{sd.loc[name, c]:.3f}"
    print(line)

print()
print("কীভাবে পড়বেন:")
print("  • noise 0% কলাম = ধাপ ২ক-এর মতো পরিষ্কার ডেটা।")
print("  • ডানে গেলে সেন্সর যত খারাপ, accuracy তত কমে — কার কমা সবচেয়ে কম, সেটাই দেখার বিষয়।")
print("  • 'Rule (fixed 450/550)' = কৃষক সীমা একটু ভুল বসালে কী হয়।")
print("    ML মডেলগুলো (NB সহ) সীমা ডেটা থেকে নিজে শেখে, তাই এই ভুলের ঝুঁকি নেই।")
print()
print("✅ সেভ হয়েছে: results\\tables\\05_noise_summary.csv")
