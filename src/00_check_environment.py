"""
ধাপ ০ — পরিবেশ পরীক্ষা (environment check)

এই স্ক্রিপ্ট কী করে:
  1. সব দরকারি লাইব্রেরি ইনস্টল আছে কিনা দেখে।
  2. কোন version আছে তা results/environment.txt-এ লিখে রাখে
     (রিভিউয়ার A-র ৭ নম্বর মন্তব্য: "software versions" দিতে হবে)।
  3. ডেটা ফাইলগুলো ঠিক জায়গায় আছে কিনা দেখে।

চালানোর নিয়ম (Predictive_Control ফোল্ডার থেকে):
    python src/00_check_environment.py
"""

import sys
import platform
from pathlib import Path

# Windows-এর কিছু টার্মিনালে বাংলা লেখা print করলে error হয় — তা ঠেকাতে UTF-8 চালু করছি
sys.stdout.reconfigure(encoding="utf-8")

# src ফোল্ডারের এক ধাপ উপরে config.py আছে — সেটা Python-কে চিনিয়ে দিচ্ছি
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# ১. লাইব্রেরি পরীক্ষা
# ---------------------------------------------------------------------------
# (import-এর নাম, pip-এর নাম)
LIBRARIES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("openpyxl", "openpyxl"),
    ("joblib", "joblib"),
]

lines = []
lines.append(f"Python: {platform.python_version()}")
lines.append(f"OS: {platform.system()} {platform.release()}")

missing = []
for import_name, pip_name in LIBRARIES:
    try:
        module = __import__(import_name)
        lines.append(f"{pip_name}: {module.__version__}")
    except ImportError:
        missing.append(pip_name)
        lines.append(f"{pip_name}: NOT INSTALLED")

print("=" * 50)
print("লাইব্রেরি ও version")
print("=" * 50)
for line in lines:
    print(" ", line)

# ---------------------------------------------------------------------------
# ২. ডেটা ফাইল পরীক্ষা
# ---------------------------------------------------------------------------
print()
print("=" * 50)
print("ডেটা ফাইল")
print("=" * 50)

data_ok = True
check_files = [config.D1_FILE] + [config.D4_DIR / f for f in config.D4_IRRIGATED_FILES]
for path in check_files:
    exists = path.exists()
    data_ok = data_ok and exists
    mark = "OK " if exists else "নেই"
    print(f"  [{mark}] {path.relative_to(config.PROJECT_ROOT)}")

# ---------------------------------------------------------------------------
# ৩. ফলাফল সেভ ও সারসংক্ষেপ
# ---------------------------------------------------------------------------
config.ensure_dirs()
env_file = config.RESULTS / "environment.txt"
env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("=" * 50)
if missing:
    print("❌ এই লাইব্রেরিগুলো নেই:", ", ".join(missing))
    print("   ইনস্টল করুন:  pip install -r requirements.txt")
elif not data_ok:
    print("⚠️  লাইব্রেরি ঠিক আছে, কিন্তু কিছু ডেটা ফাইল পাওয়া যায়নি (উপরে 'নেই' দেখুন)।")
else:
    print("✅ সব ঠিক আছে! পরের ধাপে যাওয়া যায়।")
print(f"   version-এর তালিকা সেভ হয়েছে: {env_file.relative_to(config.PROJECT_ROOT)}")
