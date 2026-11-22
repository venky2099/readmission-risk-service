"""Download the dataset from the UCI Machine Learning Repository into data/raw/diabetic_data.csv.

Run:  python scripts/fetch_data.py
Source: Clore, Cios, DeShazo and Strack (2014), Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine Learning
Repository, doi:10.24432/C5230J, licence CC BY 4.0. If the download is blocked on your network, download the zip from
https://archive.ics.uci.edu/dataset/296 in a browser and place diabetic_data.csv in data/raw/.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def find_file(zf, target):
    for name in zf.namelist():
        if name.lower().endswith(target.lower()):
            return zf.read(name)
        if name.endswith(".zip"):
            found = find_file(zipfile.ZipFile(io.BytesIO(zf.read(name))), target)
            if found:
                return found
    return None


def main():
    if config.RAW_CSV.exists():
        print("already present:", config.RAW_CSV); return
    print("downloading", config.UCI_ZIP_URL)
    data = urllib.request.urlopen(config.UCI_ZIP_URL, timeout=120).read()
    archive = zipfile.ZipFile(io.BytesIO(data))
    csv = find_file(archive, "diabetic_data.csv")
    if csv is None:
        raise SystemExit("diabetic_data.csv not found in the downloaded archive")
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    config.RAW_CSV.write_bytes(csv)
    mapping = find_file(archive, "IDs_mapping.csv")          # code meanings for admission type, discharge, source
    if mapping:
        (config.RAW_CSV.parent / "IDs_mapping.csv").write_bytes(mapping)
    print("saved", config.RAW_CSV, f"({len(csv) / 1e6:.1f} MB); src.data_pipeline checks the row and label counts")


if __name__ == "__main__":
    main()
