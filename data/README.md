# Data

The raw data are not stored in this repository. Download them and place them exactly as shown, so that `config.py` finds them:

```
data/
└── raw/
    ├── D1_autoirrigation/
    │   └── data.csv
    └── D4_bushland_cotton/
        ├── 2000_Cotton_E_Lys_ClimDat.xlsx
        ├── 2001_Cotton_E_Lys_ClimDat.xlsx
        ├── 2002_Cotton_E_Lys_ClimDat.xlsx
        ├── 2008_Cotton_E_Lys_ClimDat.xlsx
        ├── 2010_Cotton_E_Lys_ClimDat.xlsx
        ├── 2012_Cotton_W_Lys_ClimDat.xlsx
        ├── 2020_Cotton_E_Lys_ClimDat.xlsx
        ├── 2020_Cotton_W_Lys_ClimDat.xlsx
        ├── 2021_Cotton_E_Lys_ClimDat.xlsx
        └── 2021_Cotton_W_Lys_ClimDat.xlsx
```

## D1 — AutoIrrigationData (simulated)

- Source: https://www.kaggle.com/datasets/harshilpatel355/autoirrigationdata
- Download the dataset and save its `data.csv` as `data/raw/D1_autoirrigation/data.csv`.
- 200 records; columns `crop` (always cotton), `moisture` (sensor reading 0–1023, higher = drier), `temp` (°C), `pump` (0 = OFF, 1 = ON).
- Note: the pump label is perfectly separated by a single moisture value (OFF ≤ 493, ON ≥ 503), indicating rule-generated labels.

## D4 — Bushland, Texas cotton lysimeters (real field data)

- Source: Evett et al. (2022), USDA Ag Data Commons:
  https://agdatacommons.nal.usda.gov/articles/dataset/Weighing_Lysimeter_Data_for_The_Bushland_Texas_Cotton_Datasets/25114670
- Download the ten irrigated-season files listed above (the dryland "W" files of 2000, 2001, 2003, 2004 and 2008 are not used).
- Only the `Daily` sheet of each workbook is read. The label is the daily `Irrigation Token`.
- The 2000 SE lysimeter is excluded because its Irrigation Token was not recorded.

`data/processed/` is created automatically by the scripts.
