"""
config.py — পুরো প্রজেক্টের সব সেটিং এক জায়গায়।

কেন এই ফাইল?
  আগের কোডে প্রতিটা স্ক্রিপ্টে আলাদা করে লেখা ছিল  r"D:\\F\\Cotton\\Cotton\\data.csv"।
  PC বদলালেই সব ভেঙে যেত। এখন সব path এই ফাইল থেকে আসবে, আর path তৈরি হবে
  এই ফাইলের নিজের অবস্থান থেকে। ফলে ফোল্ডারটা যেকোনো ড্রাইভে/PC-তে রাখলেও চলবে।

ব্যবহার (src ফোল্ডারের যেকোনো স্ক্রিপ্টে):
    import config
    df = pd.read_csv(config.D1_FILE)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# ১. ফোল্ডারের ঠিকানা
# ---------------------------------------------------------------------------
# __file__ = এই config.py ফাইলের ঠিকানা। .parent = যে ফোল্ডারে আছে (Predictive_Control)।
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

RESULTS = PROJECT_ROOT / "results"
RESULTS_TABLES = RESULTS / "tables"        # CSV টেবিল (পেপারে বসানোর জন্য)
RESULTS_FIGURES = RESULTS / "figures"      # PNG ছবি (300 dpi)
RESULTS_MODELS = RESULTS / "models"        # সেভ করা মডেল (.pkl)

# ---------------------------------------------------------------------------
# ২. ডেটাসেট
# ---------------------------------------------------------------------------
# D1: পেপারের মূল ডেটা (Kaggle AutoIrrigationData, simulated, cotton)
D1_FILE = DATA_RAW / "D1_autoirrigation" / "data.csv"

# D4: Bushland, Texas — আসল cotton মাঠ (USDA-ARS)
D4_DIR = DATA_RAW / "D4_bushland_cotton"

# শুধু যেসব ফাইলে সেচ দেওয়া হয়েছে (dryland ফাইলে Irrigation Token সবসময় 0, তাই বাদ)
D4_IRRIGATED_FILES = [
    "2000_Cotton_E_Lys_ClimDat.xlsx",
    "2001_Cotton_E_Lys_ClimDat.xlsx",
    "2002_Cotton_E_Lys_ClimDat.xlsx",
    "2008_Cotton_E_Lys_ClimDat.xlsx",
    "2010_Cotton_E_Lys_ClimDat.xlsx",
    "2012_Cotton_W_Lys_ClimDat.xlsx",
    "2020_Cotton_E_Lys_ClimDat.xlsx",
    "2020_Cotton_W_Lys_ClimDat.xlsx",
    "2021_Cotton_E_Lys_ClimDat.xlsx",
    "2021_Cotton_W_Lys_ClimDat.xlsx",
]

# ---------------------------------------------------------------------------
# ৩. পরীক্ষার সেটিং (পেপারের Methods অংশে হুবহু এগুলোই লিখতে হবে)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42           # সব random কাজে একই seed → প্রতিবার একই ফলাফল (reproducibility)
TEST_SIZE = 0.20           # পুরনো 80:20 split আবার তৈরির জন্য
CV_FOLDS = 10              # k-fold cross-validation-এ k = 10
CV_REPEATS = 10            # পুরো 10-fold প্রক্রিয়া 10 বার, ভিন্ন ভিন্ন ভাগে → মোট 100টা ফলাফল
OUTLIER_QUANTILE = 0.99    # পেপারের মতো উপরের 1% outlier বাদ
GNB_VAR_SMOOTHING = 1e-8   # পেপারের Naive Bayes-এ ব্যবহৃত মান (NB.py থেকে)
D1_RULE_THRESHOLD = 500    # D1-এ "moisture ≥ 500 হলে পাম্প ON" নিয়ম
NOISE_LEVELS = [0.00, 0.02, 0.05, 0.10]   # sensor noise পরীক্ষা: 0%, 2%, 5%, 10%
NOISE_REPEATS = 100        # প্রতিটা noise মাত্রায় কতবার পুনরাবৃত্তি


def ensure_dirs():
    """results-এর ফোল্ডারগুলো না থাকলে বানিয়ে দেয়। প্রতিটা স্ক্রিপ্টের শুরুতে একবার ডাকা হবে।"""
    for folder in [DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS]:
        folder.mkdir(parents=True, exist_ok=True)
