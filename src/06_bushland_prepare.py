"""
ধাপ ৩ক — Bushland (আসল cotton মাঠ) ডেটা পরিষ্কার করে একটা টেবিলে আনা

কী করে:
  1. ১০টা সেচ-দেওয়া Excel ফাইলের প্রতিটা থেকে শুধু "Daily" ট্যাব পড়ে।
  2. প্রতিটা ফাইলে ২টা lysimeter (NE/SE বা NW/SW) — প্রতিটাকে আলাদা row হিসেবে নেয়।
     ফলে একটা row = একটা lysimeter-এর একটা দিন।
  3. বছরভেদে কলামের নাম একটু আলাদা (বাড়তি স্পেস, "Precip." বনাম "Precip" ইত্যাদি) —
     তাই নাম "স্বাভাবিক" করে খোঁজা হয়।
  4. সব মিলিয়ে data\\processed\\d4_bushland_daily.csv-এ সেভ করে।

কোন কলাম নেওয়া হয় (সবগুলো বছরে আছে এমন):
  • Midnight Relative Water Storage (mm) — মাটিতে কত পানি (lysimeter-এর ওজন থেকে)
  • ET from Gauge Precip (mm)            — সেদিন কত পানি বাষ্প হয়ে গেছে (evapotranspiration)
  • Gauge Precip (mm)                     — বৃষ্টি (rain gauge থেকে)
  • Air Temp (°C), RH (%), Wind (m/s)    — আবহাওয়া
  • Irrigation / Maint. / Missing Data Token — label আর মান-নিয়ন্ত্রণ

ইচ্ছা করে বাদ দেওয়া:
  • "Catch Precip" — lysimeter-এর ওপর জমা পানি, এতে সেচের পানিও ধরা পড়ে → label ফাঁস (leakage)
  • Rs (সৌর বিকিরণ) — 2020_W ফাইলে নেই, তাই সব বছরে তুলনা করা যায় না

চালানো:  python src\\06_bushland_prepare.py   (প্রায় ৩০ সেকেন্ড)
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

warnings.filterwarnings("ignore")
config.ensure_dirs()


def norm(s):
    """কলামের নাম ছোট হাতের অক্ষরে, স্পেস আর ডট বাদ দিয়ে — যাতে তুলনা সহজ হয়।"""
    return re.sub(r"[\s\.]+", "", str(s).lower())


def find_col(columns, prefix, must, must_not=()):
    """
    prefix (যেমন 'ne') দিয়ে শুরু এবং must-এর সব শব্দ আছে, must_not-এর কোনোটা নেই — এমন কলাম খোঁজে।
    একাধিক মিললে প্রথমটা নেয়। না পেলে None।
    """
    for c in columns:
        n = norm(c)
        if n.startswith(prefix) and all(m in n for m in must) and not any(x in n for x in must_not):
            return c
    return None


def daily_sheet_name(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    names = [s for s in wb.sheetnames if re.search(r"Daily\s*$", s) and not s.lower().startswith("dic")]
    wb.close()
    return names[0]


all_rows = []
for fname in config.D4_IRRIGATED_FILES:
    path = config.D4_DIR / fname
    sheet = daily_sheet_name(path)
    df = pd.read_excel(path, sheet_name=sheet, na_values=["#N/A", "NA", "#VALUE!", ""])
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]          # অপ্রয়োজনীয় row বাদ
    cols = list(df.columns)

    side = "E" if "_E_" in fname else "W"
    lysimeters = ["ne", "se"] if side == "E" else ["nw", "sw"]

    for lys in lysimeters:
        spec = {
            "storage_mm": find_col(cols, lys, ["midnight", "storage"]),
            "et_mm":      find_col(cols, lys, ["etfrom", "gauge"]) or find_col(cols, lys, ["etfromgage"]),
            "precip_mm":  find_col(cols, lys, ["gauge", "precip"], ["etfrom"]),
            "air_temp_c": find_col(cols, lys, ["airtemp"]),
            "rh_pct":     find_col(cols, lys, ["rh", "%"]) or find_col(cols, lys, ["rhin"]),
            "wind_ms":    find_col(cols, lys, ["windspeed"]),
            "irrigation": find_col(cols, lys, ["irrigation", "token"]),
            "maint":      find_col(cols, lys, ["maint", "token"]),
            "missing":    find_col(cols, lys, ["missing", "token"]),
        }
        out = pd.DataFrame({
            "year": df["Year"].astype(int).to_numpy(),
            "doy": pd.to_numeric(df["DOY"], errors="coerce").astype(int).to_numpy(),
            "lysimeter": f"{df['Year'].iloc[0]}_{lys.upper()}",
            "source_file": fname,
        })
        for new, old in spec.items():
            if old is None:
                # 2000 সালের ফাইলে Maint./Missing token নেই — 0 ধরা হলো (ঘটনা ছিল না বা লেখা হয়নি)
                out[new] = 0 if new in ("maint", "missing") else np.nan
            else:
                out[new] = pd.to_numeric(df[old], errors="coerce").to_numpy()
        missing_cols = [k for k, v in spec.items() if v is None]
        print(f"  {fname:<32} {lys.upper()}: {len(out)} দিন, সেচের দিন = {int(out.irrigation.sum())}"
              + (f"  (নেই: {missing_cols})" if missing_cols else ""))
        all_rows.append(out)

data = pd.concat(all_rows, ignore_index=True)
for c in ["irrigation", "maint", "missing"]:
    data[c] = data[c].fillna(0).astype(int)

out_path = config.DATA_PROCESSED / "d4_bushland_daily.csv"
data.to_csv(out_path, index=False, float_format="%.4f")

print()
print("=" * 60)
print(f"মোট: {len(data)} row ({data.lysimeter.nunique()}টা lysimeter-মৌসুম, "
      f"{data.year.nunique()}টা বছর)")
print(f"সেচের দিন: {data.irrigation.sum()} ({data.irrigation.mean() * 100:.1f}%)")
print(f"✅ সেভ হয়েছে: {out_path.relative_to(config.PROJECT_ROOT)}")
