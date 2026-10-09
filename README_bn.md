# Predictive_Control — পেপার রিভিশনের পুরো কাজ

**পেপার:** Predictive Control of Agricultural Irrigation Using Machine Learning: A Naive Bayes-Based IoT Framework for Cotton Crops
**জার্নাল:** IIUC Studies (Paper ID: BE-0306) — সিদ্ধান্ত: Major Revision

এই ফোল্ডারের সব কিছু রিভিশনের জন্য নতুন করে বানানো। আগের কাজ (`D:\Cotton\NB.py`, `Comparison.py`, `output\`) যেমন ছিল তেমনই আছে, সেগুলোতে হাত দেওয়া হয়নি।

---

## ফোল্ডার কাঠামো

```
Predictive_Control\
├── README.md            ← এই ফাইল
├── config.py            ← সব path ও সেটিং এক জায়গায় (PC বদলালেও কোড চলবে)
├── requirements.txt     ← কোন লাইব্রেরি লাগবে
├── data\
│   ├── raw\             ← আসল ডেটা — কখনো হাতে বদলাবেন না
│   │   ├── D1_autoirrigation\    ← মূল ডেটা (Kaggle, simulated cotton, 200 row)
│   │   ├── D4_bushland_cotton\   ← যাচাইয়ের ডেটা (USDA, আসল cotton মাঠ)
│   │   ├── D2_mendeley_iot\      ← (ব্যবহার করা হচ্ছে না — রেখে দেওয়া হয়েছে)
│   │   └── D3_kaggle_multicrop\  ← (ব্যবহার করা হচ্ছে না — রেখে দেওয়া হয়েছে)
│   └── processed\       ← কোড দিয়ে পরিষ্কার করা ডেটা
├── src\                 ← সব কোড, নম্বর অনুযায়ী ক্রমে চালাতে হবে
├── results\
│   ├── tables\          ← ফলাফলের টেবিল (CSV)
│   ├── figures\         ← ছবি (PNG, 300 dpi)
│   └── models\          ← সেভ করা মডেল
└── paper\               ← রিভিশন চেকলিস্ট, Response to Reviewers
```

---

## প্রথমবার সেটআপ (একবারই করতে হবে)

Windows-এ **Command Prompt** খুলুন, তারপর একটা একটা লাইন চালান:

```bat
:: ১. প্রজেক্ট ফোল্ডারে যান
cd /d D:\Cotton\Predictive_Control

:: ২. এই প্রজেক্টের জন্য আলাদা Python পরিবেশ (virtual environment) বানান
python -m venv .venv

:: ৩. পরিবেশটা চালু করুন (লাইনের শুরুতে (.venv) দেখা যাবে)
.venv\Scripts\activate

:: ৪. লাইব্রেরি ইনস্টল করুন (ইন্টারনেট লাগবে, কয়েক মিনিট সময় নেবে)
python -m pip install --upgrade pip
pip install -r requirements.txt

:: ৫. সব ঠিক আছে কিনা পরীক্ষা করুন
python src\00_check_environment.py
```

শেষে **"✅ সব ঠিক আছে!"** দেখা গেলে সেটআপ শেষ।

> **পরে প্রতিবার কাজ শুরুর আগে** শুধু ধাপ ১ আর ৩ করলেই হবে।
> PyCharm ব্যবহার করলে: File → Open → `Predictive_Control` → Settings → Python Interpreter → `.venv` বেছে নিন।

---

## কোন স্ক্রিপ্ট কী করে (ধাপে ধাপে যোগ হবে)

| স্ক্রিপ্ট | কাজ | রিভিউয়ারের কোন মন্তব্য | অবস্থা |
|---|---|---|---|
| `00_check_environment.py` | লাইব্রেরি, version ও ডেটা ফাইল পরীক্ষা | A-7 (reproducibility) | ✅ |
| `01_reproduce_original.py` | পেপারের পুরনো ফলাফল (97%, confusion matrix 11/1/0/27) হুবহু আবার বের করা | ভিত্তি (baseline) | ✅ |
| `utils.py` | সাধারণ কাজ: ডেটা লোড, outlier বাদ (শুধু train থেকে), threshold rule, সব মডেল | — | (সরাসরি চালাতে হয় না) |
| `02_cross_validation.py` | 10×10-fold CV, ৭টা মডেলের তুলনা (threshold rule সহ) | A-2, A-3, A-5, B | ✅ |
| `03_ablation.py` | NB-র ৮টা configuration: raw/log/দুটো, outlier বাদ/না, crop, temp | A-4, A-5 | ✅ → **C4 (শুধু log) বাছাই** |
| `04_iot_benchmark.py` | Training সময়, একটা রিডিং-এ অনুমানের সময়, মডেলের আকার, সংরক্ষিত সংখ্যা | A-6 | ✅ |
| `05_noise_robustness.py` | Sensor noise (0–10%) আর ভুল হাতে-বসানো সীমায় কে কতটা টেকে | A-1, A-3 | ✅ |
| `06_bushland_prepare.py` | Bushland-এর ১০টা Excel থেকে Daily ডেটা এক টেবিলে (৭,৩১০ দিন) | A-1, B | ✅ |
| `07_bushland_models.py` | আসল cotton মাঠে সেচ-সিদ্ধান্ত অনুমান, leave-one-year-out, ৮টা মডেল | A-1, A-2, A-3, B | ✅ |
| `08_make_figures.py` | পেপারের ৮টা ছবি (300 dpi) + প্রস্তাবিত caption | A minor-3 | ⏳ |

> **নোট (৪ অক্টোবর):** C4 বাছাইয়ের পর `utils.py`-তে "Gaussian NB" এখন শুধু log feature ব্যবহার করে।
> তাই `02_cross_validation.py` আবার চালাতে হবে (NB: 0.984 ± 0.026)।

---

## রিভিশনের পরিকল্পনা (সংক্ষেপে)

1. **ধাপ ০** — সেটআপ ✅
2. **ধাপ ১** — পুরনো ফলাফল (97%) আবার বের করা
3. **ধাপ ২** — Cross-validation, মডেল তুলনা (threshold rule সহ), ablation, IoT benchmark, noise পরীক্ষা
4. **ধাপ ৩** — Bushland আসল cotton ডেটায় যাচাই
5. **ধাপ ৪** — ছবি ও টেবিল
6. **ধাপ ৫** — পেপার রিভিশন চেকলিস্ট ও Response to Reviewers

## মূল নিয়ম
- `data\raw`-এর কোনো ফাইল হাতে বদলাবেন না।
- `results\` আর `data\processed\`-এর সব কিছু কোড আবার চালালেই তৈরি হয় — দরকারে মুছে আবার চালানো যায়।
- কোডে কখনো `D:\...` লিখবেন না — সব path `config.py` থেকে নিন।
