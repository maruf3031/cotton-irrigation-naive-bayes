# Predictive Control of Agricultural Irrigation Using Machine Learning: A Naive Bayes-Based IoT Framework for Cotton Crops

Analysis code for the revised manuscript **BE-0306** (*IIUC Studies*).

**Authors:** Nahid Sultana, Sultana Akter (corresponding), Mohammad Shyfur Rahman Chowdhury, Md. Maruf Hasan

This repository contains every script needed to reproduce all tables and figures in the revised paper, from the raw public datasets. Code comments are written in Bengali; a Bengali guide is available in [`README_bn.md`](README_bn.md).

---

## What the study does

A Gaussian Naive Bayes (GNB) classifier decides whether an irrigation pump should be ON or OFF from soil moisture and air temperature. The revised analysis:

1. compares GNB with a soil-moisture threshold rule, logistic regression, decision tree, k-nearest neighbours, SVM and random forest using 10 × 10 repeated stratified cross-validation;
2. runs an ablation study of the preprocessing steps;
3. tests robustness to simulated sensor noise and to a wrongly set manual threshold;
4. benchmarks the computational cost of each model for IoT deployment;
5. validates all models externally on real cotton-field lysimeter data from USDA-ARS Bushland, Texas, using leave-one-year-out evaluation.

### Main results

| | Gaussian NB | Best comparator |
|---|---|---|
| Simulated data (D1), accuracy, 100 test folds | 0.984 ± 0.026 | 1.000 (threshold rule; D1 labels are rule-based) |
| Real cotton field (Bushland), ROC-AUC, 8 held-out seasons | 0.680 ± 0.070 | 0.704 ± 0.089 (random forest) |
| Real cotton field, balanced accuracy | 0.631 ± 0.044 | 0.653 ± 0.088 (SVM) |
| Stored model parameters | 10 (1.44 KB) | — |

---

## Repository structure

```
├── config.py              # all paths and experimental settings (seed = 42)
├── requirements.txt       # Python dependencies
├── src/
│   ├── utils.py                    # shared data loading, outlier trimming, models
│   ├── 00_check_environment.py     # checks libraries, versions and data files
│   ├── 01_reproduce_original.py    # reproduces the original single-split result (97%)
│   ├── 02_cross_validation.py      # 10×10 repeated stratified CV, 7 models
│   ├── 03_ablation.py              # 8 preprocessing configurations
│   ├── 04_iot_benchmark.py         # latency, model size, stored parameters
│   ├── 05_noise_robustness.py      # sensor-noise and wrong-threshold test
│   ├── 06_bushland_prepare.py      # builds the daily field table from the Excel files
│   ├── 07_bushland_models.py       # external validation, leave-one-year-out
│   └── 08_make_figures.py          # all figures (300 dpi) and draft captions
├── results/
│   ├── environment.txt    # software versions used for the paper
│   ├── tables/            # every result table (CSV)
│   └── figures/           # Figs. 1–8 and figure_captions.txt
└── data/README.md         # where to download the datasets
```

---

## How to reproduce

### 1. Get the data

The datasets are **not** included in this repository. Download them as described in [`data/README.md`](data/README.md).

### 2. Install (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\00_check_environment.py
```

On Linux/macOS, use `source .venv/bin/activate` and forward slashes.

### 3. Run all scripts in order

```bat
python src\01_reproduce_original.py
python src\02_cross_validation.py
python src\03_ablation.py
python src\04_iot_benchmark.py
python src\05_noise_robustness.py
python src\06_bushland_prepare.py
python src\07_bushland_models.py
python src\08_make_figures.py
```

Total running time is about 5 minutes on a desktop PC. Timing values in `04_iot_benchmark.py` depend on the machine.

### Software used for the paper

Python 3.12.10, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, Matplotlib 3.11.2, seaborn 0.13.2, openpyxl 3.1.5, joblib 1.6.0 (Windows 11).

---

## Data sources

- **D1 — AutoIrrigationData** (simulated, 200 records): Patel, H. (2020). Kaggle. https://www.kaggle.com/datasets/harshilpatel355/autoirrigationdata
- **D4 — Bushland, Texas cotton lysimeters** (real field data): Evett, S. R., Marek, G. W., Copeland, K. S., Howell, T. A. Sr., Colaizzi, P. D., Brauer, D. K., & Ruthardt, B. B. (2022). *Weighing Lysimeter Data for The Bushland, Texas, Cotton Datasets*. Ag Data Commons. https://agdatacommons.nal.usda.gov/articles/dataset/Weighing_Lysimeter_Data_for_The_Bushland_Texas_Cotton_Datasets/25114670

---

## Citation

If you use this code, please cite the paper (citation details will be added after publication) and this repository; see [`CITATION.cff`](CITATION.cff).

## License

Code: [MIT License](LICENSE). The datasets remain under the terms of their original providers.
