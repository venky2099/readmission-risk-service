"""Latency and throughput benchmark of the real HTTP service.

Run:  python -m serving.benchmark [--n 1000]
Starts uvicorn as a separate process on 127.0.0.1:8765, waits for /health, sends 50 warm-up requests, then
N sequential requests with real test-set encounters, then the same number from 8 concurrent clients.
Latency is measured client-side (time from sending a request to receiving the full response).
Also measures the model alone, in-process, to separate model time from HTTP and validation overhead.
Writes reports/benchmark.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import joblib
import numpy as np
import pandas as pd

from serving.payloads import row_to_payload
from src import config
from src.train import load_parts

URL = "http://127.0.0.1:8765"


def pct(a):
    a = np.asarray(a)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean()), "max_ms": float(a.max())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); n = ap.parse_args().n
    te = load_parts(("test",))["test"]
    te = te[~te.discharge_disposition_id.isin(config.EXCLUDED_DISPOSITIONS)]
    payloads = [row_to_payload(r) for r in te.sample(n, random_state=config.SEED).to_dict("records")]

    def start(workers):
        env = {**os.environ, "LOG_LEVEL": "WARNING"}
        srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", "8765",
                                "--workers", str(workers), "--log-level", "warning"], cwd=str(config.ROOT), env=env)
        with httpx.Client(timeout=10) as c:
            for _ in range(240):
                try:
                    if c.get(f"{URL}/health").status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(0.5)
            else:
                srv.terminate(); raise RuntimeError("service did not start")
            for p in payloads[:50]:
                c.post(f"{URL}/predict", json=p)          # warm-up
        return srv

    def sequential():
        lat, errors = [], 0
        with httpx.Client(timeout=10) as c:
            t_all = time.perf_counter()
            for p in payloads:
                t0 = time.perf_counter(); r = c.post(f"{URL}/predict", json=p); lat.append((time.perf_counter() - t0) * 1000)
                errors += r.status_code != 200
            wall = time.perf_counter() - t_all
        return {**pct(lat), "throughput_rps": n / wall, "errors": int(errors)}

    def concurrent(clients=8):
        def worker(chunk):
            out, err = [], 0
            with httpx.Client(timeout=30) as cc:
                for p in chunk:
                    t0 = time.perf_counter(); r = cc.post(f"{URL}/predict", json=p); out.append((time.perf_counter() - t0) * 1000)
                    err += r.status_code != 200
            return out, err
        t_all = time.perf_counter()
        with ThreadPoolExecutor(clients) as ex:
            res = list(ex.map(worker, [payloads[i::clients] for i in range(clients)]))
        wall = time.perf_counter() - t_all
        return {**pct([x for r in res for x in r[0]]), "throughput_rps": n / wall, "errors": int(sum(r[1] for r in res))}

    scenarios = {}
    server = start(1)
    try:
        scenarios["1 worker, 1 client (sequential)"] = sequential()
        scenarios["1 worker, 8 concurrent clients"] = concurrent()
        server_metrics = httpx.get(f"{URL}/metrics").json()
    finally:
        server.terminate(); server.wait(timeout=30)
    time.sleep(2)
    server = start(4)
    try:
        scenarios["4 workers, 8 concurrent clients"] = concurrent()
    finally:
        server.terminate(); server.wait(timeout=30)

    model = joblib.load(config.MODEL_PATH)
    frame = pd.DataFrame(payloads[:300])
    model.predict_proba(frame.head(5))
    alone = []
    for i in range(300):
        t0 = time.perf_counter(); model.predict_proba(frame.iloc[[i]]); alone.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter(); model.predict_proba(pd.DataFrame(payloads)); batch_s = time.perf_counter() - t0

    out = {"machine": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
                       "logical_cpus": os.cpu_count()},
           "n_requests": n, "http": scenarios,
           "model_only_single_row": pct(alone),
           "model_only_batch": {"rows": n, "seconds": batch_s, "rows_per_second": n / batch_s},
           "server_side_metrics": server_metrics}
    (config.REPORTS / "benchmark.json").write_text(json.dumps(out, indent=1))
    for name, s in scenarios.items():
        print(f"{name:34s} p50 {s['p50_ms']:6.1f} ms  p95 {s['p95_ms']:6.1f}  p99 {s['p99_ms']:6.1f}  "
              f"{s['throughput_rps']:5.0f} req/s  errors {s['errors']}")
    m = out["model_only_single_row"]
    print(f"model only: p50 {m['p50_ms']:.1f} ms per row; batch {out['model_only_batch']['rows_per_second']:.0f} rows/s")


if __name__ == "__main__":
    main()
