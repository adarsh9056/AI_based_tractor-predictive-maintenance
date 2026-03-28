"""
Shared sensor feature engineering (AI4I-style).

Mechanical power uses the standard approximation P[kW] = (T[Nm] * n[rpm]) / 9549.
PWF rule-of-thumb band is 3.5–9 kW, equivalent to the original 3500–9000 W spec.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PWF_KW_MIN = 3.5
PWF_KW_MAX = 9.0
HDF_TEMP_GAP_K = 8.6
HDF_RPM_THRESHOLD = 1380.0
OSF_WT_MIN = 11000.0
OSF_WT_MAX = 13000.0
TWF_WEAR_MIN = 200.0
TWF_WEAR_MAX = 240.0

FEATURE_LIST = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power_kw",
    "wear_torque",
    "hdf_risk",
    "pwf_risk",
    "osf_risk",
    "twf_band",
]


def power_kw(rpm: float, torque: float) -> float:
    return (rpm * torque) / 9549.0


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_diff"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["power_kw"] = power_kw(out["Rotational speed [rpm]"], out["Torque [Nm]"])
    out["wear_torque"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    out["hdf_risk"] = ((out["temp_diff"] < HDF_TEMP_GAP_K) & (out["Rotational speed [rpm]"] < HDF_RPM_THRESHOLD)).astype(int)
    out["pwf_risk"] = ((out["power_kw"] < PWF_KW_MIN) | (out["power_kw"] > PWF_KW_MAX)).astype(int)
    out["osf_risk"] = ((out["wear_torque"] >= OSF_WT_MIN) & (out["wear_torque"] <= OSF_WT_MAX)).astype(int)
    out["twf_band"] = ((out["Tool wear [min]"] >= TWF_WEAR_MIN) & (out["Tool wear [min]"] <= TWF_WEAR_MAX)).astype(int)
    return out


def build_feature_row(
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
) -> dict[str, Any]:
    temp_diff = process_temp - air_temp
    pkw = power_kw(rpm, torque)
    wear_torque = tool_wear * torque
    hdf_risk = int(temp_diff < HDF_TEMP_GAP_K and rpm < HDF_RPM_THRESHOLD)
    pwf_risk = int(pkw < PWF_KW_MIN or pkw > PWF_KW_MAX)
    osf_risk = int(OSF_WT_MIN <= wear_torque <= OSF_WT_MAX)
    twf_band = int(TWF_WEAR_MIN <= tool_wear <= TWF_WEAR_MAX)
    return {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "temp_diff": temp_diff,
        "power_kw": pkw,
        "wear_torque": wear_torque,
        "hdf_risk": hdf_risk,
        "pwf_risk": pwf_risk,
        "osf_risk": osf_risk,
        "twf_band": twf_band,
    }


def row_to_vector(row: dict[str, Any], feature_order: list[str] | None = None) -> np.ndarray:
    order = feature_order or FEATURE_LIST
    return np.array([[float(row[k]) for k in order]], dtype=np.float64)


def explanation_factors(row: dict[str, Any]) -> list[str]:
    """Human-readable rule-based hints (not SHAP)."""
    reasons: list[str] = []
    if row["hdf_risk"]:
        reasons.append(
            f"Low temperature gap (ΔT = {row['temp_diff']:.2f} K < {HDF_TEMP_GAP_K} K) with RPM "
            f"below {HDF_RPM_THRESHOLD:.0f} elevates heat-dissipation (HDF) risk per AI4I-style rules."
        )
    if row["pwf_risk"]:
        reasons.append(
            f"Mechanical power {row['power_kw']:.2f} kW is outside the normal band "
            f"({PWF_KW_MIN}–{PWF_KW_MAX} kW), matching power-failure (PWF) heuristics."
        )
    if row["osf_risk"]:
        reasons.append(
            f"Wear × torque product ({row['wear_torque']:.0f} min·Nm) lies in the {OSF_WT_MIN:.0f}–{OSF_WT_MAX:.0f} "
            "window associated with overstrain (OSF)."
        )
    if row["twf_band"]:
        wear = row["Tool wear [min]"]
        reasons.append(
            f"Component wear time {wear:.0f} min falls in the {TWF_WEAR_MIN:.0f}–{TWF_WEAR_MAX:.0f} min "
            "replacement window (TWF)."
        )
    if not reasons:
        reasons.append("No rule-based risk flags fired; model relies on full feature vector including raw sensors.")
    return reasons
