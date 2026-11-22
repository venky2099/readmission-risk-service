"""Shared settings for the readmission risk service. Every script imports from here, so one change applies everywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "diabetic_data.csv"
SPLIT_CSV = ROOT / "data" / "processed" / "split.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
METADATA_PATH = ARTIFACTS / "metadata.json"
REFERENCE_PATH = ARTIFACTS / "drift_reference.json"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42

# Official source (UCI Machine Learning Repository, CC BY 4.0, doi:10.24432/C5230J).
UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/296/"
               "diabetes+130-us+hospitals+for+years+1999-2008.zip")
EXPECTED_ROWS = 101_766
EXPECTED_PATIENTS = 71_518
EXPECTED_LT30 = 11_357

# Discharge dispositions after which a 30-day readmission is impossible (death) or not the goal of a follow-up
# call (hospice). In the data, codes 11, 19 and 20 have zero readmissions of any kind; 21 does not occur. Codes 13
# and 14 (hospice) are removed as in Strack et al. (2014).
EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

# Capacity rule (my decision): the follow-up team can call 20 of every 100 eligible discharges.
CAPACITY = 0.20

# Patient-grouped split: about 70% train, 15% validation, 15% test, no patient in two parts.
N_SPLIT_FOLDS = 20          # 14 folds train, 3 validation, 3 test
VAL_FOLDS = (14, 15, 16)
TEST_FOLDS = (17, 18, 19)

MODEL_VERSION = "1.0.0"
