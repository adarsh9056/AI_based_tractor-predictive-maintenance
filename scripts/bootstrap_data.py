#!/usr/bin/env python3
"""
If ai4i2020.csv is missing, generate a synthetic dataset matching AI4I 2020 schema.
Replace with the official Kaggle CSV when available:
https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COLS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def _power_w(rpm: float, torque: float) -> float:
    return (rpm * torque) / 9549.0


def generate(n: int = 10_000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        typ = RNG.choice(["L", "M", "H"], p=[0.6, 0.3, 0.1])
        twf = hdf = pwf = osf = rnf = 0

        u = RNG.random()
        if u < 0.001:
            rnf = 1
            air = float(RNG.uniform(296.0, 304.0))
            proc = air + float(RNG.uniform(6.0, 14.0))
            rpm = float(RNG.uniform(1200, 2000))
            torque = float(RNG.uniform(20, 50))
            wear = float(RNG.uniform(0, 199))
        elif u < 0.04:
            twf = 1
            air = float(RNG.uniform(296.0, 304.0))
            proc = air + float(RNG.uniform(8.0, 14.0))
            rpm = float(RNG.uniform(1300, 2000))
            torque = float(RNG.uniform(20, 50))
            wear = float(RNG.uniform(200.0, 240.0))
        elif u < 0.10:
            hdf = 1
            air = float(RNG.uniform(298.0, 304.0))
            proc = air + float(RNG.uniform(4.0, 8.5))
            rpm = float(RNG.uniform(1000, 1379.0))
            torque = float(RNG.uniform(18, 45))
            wear = float(RNG.uniform(0, 199))
        elif u < 0.22:
            pwf = 1
            air = float(RNG.uniform(296.0, 304.0))
            proc = air + float(RNG.uniform(8.0, 14.0))
            if RNG.random() < 0.5:
                rpm = float(RNG.uniform(1000, 1350))
                torque = float(RNG.uniform(10, 22))
            else:
                rpm = float(RNG.uniform(1900, 2400))
                torque = float(RNG.uniform(40, 60))
            wear = float(RNG.uniform(0, 199))
        elif u < 0.30:
            osf = 1
            torque = float(RNG.uniform(35, 55))
            wear = float(RNG.uniform(210, 250))
            if wear * torque < 11000:
                wear = 11000.0 / max(torque, 1.0) + float(RNG.uniform(0, 20))
            if wear * torque > 13000:
                wear = 12500.0 / max(torque, 1.0)
            air = float(RNG.uniform(296.0, 304.0))
            proc = air + float(RNG.uniform(8.0, 14.0))
            rpm = float(RNG.uniform(1400, 2100))
        else:
            air = float(RNG.uniform(296.0, 304.0))
            proc = air + float(RNG.uniform(9.0, 15.0))
            rpm = float(RNG.uniform(1380, 2200))
            torque = float(RNG.uniform(22, 48))
            wear = float(RNG.uniform(0, 190))
            pwr = _power_w(rpm, torque)
            td = proc - air
            wt = wear * torque
            if td < 8.6 and rpm < 1380:
                proc = air + float(RNG.uniform(9.5, 15.0))
            if pwr < 3.5 or pwr > 9.0:
                rpm = float(RNG.uniform(1500, 2000))
                torque = float(RNG.uniform(25, 45))
            if 11000 <= wt <= 13000:
                wear = float(RNG.uniform(0, min(180.0, 10999.0 / max(torque, 1.0))))

        mf = 1 if any([twf, hdf, pwf, osf, rnf]) else 0
        rows.append(
            {
                "UDI": i + 1,
                "Product ID": f"M{i % 100:04d}",
                "Type": typ,
                "Air temperature [K]": round(air, 2),
                "Process temperature [K]": round(proc, 2),
                "Rotational speed [rpm]": round(rpm, 2),
                "Torque [Nm]": round(torque, 2),
                "Tool wear [min]": round(wear, 2),
                "Machine failure": mf,
                "TWF": twf,
                "HDF": hdf,
                "PWF": pwf,
                "OSF": osf,
                "RNF": rnf,
            }
        )
    return pd.DataFrame(rows, columns=COLS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv")
    p.add_argument("--n", type=int, default=10_000)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
