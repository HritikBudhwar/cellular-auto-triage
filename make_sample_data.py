"""Generate small synthetic traces in the same CSV schema as the UCC 5G dataset.

Used only for the CI smoke test so the pipeline can be checked without downloading the real data.
Never report numbers from this synthetic data.
"""
import os
import sys

import numpy as np
import pandas as pd


def make_trace(rng, n, mobility):
    t0 = pd.Timestamp("2026-01-01 10:00:00")
    rsrp = -85 + np.cumsum(rng.normal(0, 0.8, n)).clip(-25, 15)
    snr = 12 + np.cumsum(rng.normal(0, 0.5, n)).clip(-12, 10)
    for _ in range(n // 120):  # inject coverage / interference dips
        s = rng.integers(0, n - 20)
        rsrp[s:s + 15] -= rng.uniform(10, 25)
        snr[s:s + 15] -= rng.uniform(5, 12)
    cqi = np.clip(np.round(7 + snr / 3 + rng.normal(0, 1, n)), 1, 15)
    cell = np.cumsum(rng.random(n) < (0.02 if mobility == "Driving" else 0.002))
    cap = np.exp((rsrp + 120) / 12) * np.exp(snr / 15) * 400
    dl = np.maximum(cap * rng.uniform(0.6, 1.1, n), 0)
    dl[rng.random(n) < 0.03] *= 0.2  # congestion drops
    return pd.DataFrame({
        "Timestamp": [(t0 + pd.Timedelta(seconds=i)).strftime("%Y.%m.%d_%H.%M.%S") for i in range(n)],
        "Longitude": 0, "Latitude": 0,
        "Speed": rng.uniform(20, 80, n).round() if mobility == "Driving" else 0,
        "Operatorname": "A", "CellID": cell, "NetworkMode": "5G",
        "RSRP": rsrp.round(), "RSRQ": (-10 + rng.normal(0, 2, n)).round(), "SNR": snr.round(1),
        "CQI": cqi, "RSSI": (rsrp + 20).round(), "DL_bitrate": dl.round(), "UL_bitrate": (dl * 0.1).round(),
        "State": "D",
    })


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    rng = np.random.default_rng(0)
    for mob in ["Static", "Driving"]:
        os.makedirs(os.path.join(out, mob), exist_ok=True)
        for k in range(4):
            make_trace(rng, 900, mob).to_csv(os.path.join(out, mob, f"trace_{k}.csv"), index=False)
    print(f"wrote synthetic traces to {out}/")
