"""Reproduce every result in the report with one command.

Run:  python run_all.py
Order matters: the test set is read only by src.evaluate and later steps, after the model and threshold are saved.
Writes reports/run_log.json with the time of each step and the package versions used.
"""
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version

from src import config

STEPS = [
    ("fetch data", [sys.executable, "scripts/fetch_data.py"]),
    ("clean and split", [sys.executable, "-m", "src.data_pipeline"]),
    ("train and tune (train + validation only)", [sys.executable, "-m", "src.train"]),
    ("single test evaluation", [sys.executable, "-m", "src.evaluate"]),
    ("drift monitoring", [sys.executable, "-m", "src.monitor"]),
    ("robustness checks", [sys.executable, "-m", "serving.robustness"]),
    ("latency benchmark", [sys.executable, "-m", "serving.benchmark", "--n", "1000"]),
    ("unit tests", [sys.executable, "-m", "pytest", "-q"]),
]


def main():
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(), "platform": platform.platform(),
           "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "fastapi", "pydantic",
                                                "uvicorn", "httpx", "matplotlib", "pytest"]}, "steps": []}
    for name, cmd in STEPS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(config.ROOT))
        log["steps"].append({"step": name, "seconds": round(time.time() - t0, 1), "return_code": rc})
        if rc != 0:
            log["failed_at"] = name
            break
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "run_log.json").write_text(json.dumps(log, indent=1))
    print("\n" + "\n".join(f"{s['step']:45s} {s['seconds']:7.1f}s  rc={s['return_code']}" for s in log["steps"]))
    sys.exit(1 if "failed_at" in log else 0)


if __name__ == "__main__":
    main()
