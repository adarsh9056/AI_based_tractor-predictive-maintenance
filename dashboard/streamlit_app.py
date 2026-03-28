"""Streamlit UI for tractor predictive maintenance (calls FastAPI)."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("TRACTOR_API_URL", "http://127.0.0.1:8000")


def health_score(status: str, failure_p: float) -> float:
    if status == "Healthy":
        return max(35.0, 100.0 * (1.0 - failure_p))
    if status == "Warning":
        return max(20.0, 55.0 * (1.0 - failure_p))
    return max(5.0, 25.0 * (1.0 - failure_p))


def k_to_c(k: float) -> float:
    return k - 273.15


st.set_page_config(page_title="Tractor Predictive Maintenance", layout="wide")
st.title("Tractor Predictive Maintenance")
st.caption("Sensor-driven failure classification with optional LLM field guidance.")

with st.sidebar:
    st.subheader("API")
    api_url = st.text_input("Backend URL", value=API)
    use_llm = st.toggle("Include LLM diagnosis", value=True)
    st.markdown("Start API: `cd backend && uvicorn app:app --reload`")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sensor input (Kelvin in API)")
    air = st.slider("Ambient temperature [K]", 295.0, 305.0, 298.0, 0.1)
    proc = st.slider("Machine operating temperature [K]", 300.0, 315.0, 308.0, 0.1)
    rpm = st.slider("Rotational speed [rpm]", 1000.0, 2500.0, 1500.0, 10.0)
    st.caption(f"Ambient ≈ {k_to_c(air):.1f} °C · Machine ≈ {k_to_c(proc):.1f} °C")
with col2:
    torque = st.slider("Torque [Nm]", 10.0, 60.0, 35.0, 0.5)
    wear = st.slider("Component wear time [min]", 0.0, 260.0, 120.0, 1.0)
    temp_diff = proc - air
    power_kw = (rpm * torque) / 9549.0
    st.metric("ΔT (K)", f"{temp_diff:.2f}")
    st.metric("Mechanical power (kW)", f"{power_kw:.2f}", help="P_kW = T×n/9549; PWF band ~3.5–9 kW")

payload = {
    "air_temp": air,
    "process_temp": proc,
    "rpm": rpm,
    "torque": torque,
    "tool_wear": wear,
}

if st.button("Run prediction", type="primary"):
    path = "/predict_with_diagnosis" if use_llm else "/predict"
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{api_url.rstrip('/')}{path}", json=payload)
        r.raise_for_status()
        res = r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error: {exc}")
        st.stop()

    pred = res.get("prediction", "—")
    band = res.get("health_status", "—")
    fp = float(res.get("failure_probability", 0.0))
    score = health_score(band, fp)

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", pred)
    c2.metric("Health band", band)
    c3.metric("Health score (heuristic %)", f"{score:.1f}")

    st.progress(min(1.0, fp), text=f"Failure probability (1 − P(Healthy)): {fp:.1%}")

    factors = res.get("explanation_factors") or []
    if factors:
        st.subheader("Rule-based hints")
        for f in factors:
            st.markdown(f"- {f}")

    probs = res.get("class_probabilities")
    if probs:
        st.subheader("Class probabilities")
        st.bar_chart(pd.Series(probs).sort_values(ascending=False))

    if use_llm and res.get("llm_diagnosis"):
        st.subheader("Diagnosis text")
        st.markdown(res["llm_diagnosis"])
    cites = res.get("manual_citations") or []
    if cites:
        st.subheader("Manual citations")
        for i, c in enumerate(cites, 1):
            st.markdown(f"**[{i}]** {c.get('diagnosis', '')} — _score {c.get('match_score', '')}_")

st.divider()
st.subheader("Log analytics & trends")
try:
    with httpx.Client(timeout=15.0) as client:
        lr = client.get(f"{api_url.rstrip('/')}/logs", params={"limit": 200})
    lr.raise_for_status()
    data = lr.json()
    items = data.get("items", [])
    total = data.get("total", len(items))
    st.caption(f"Showing up to {len(items)} rows (total in DB filter scope: {total}).")
    if items:
        df = pd.DataFrame(items)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Failure distribution (recent)**")
            if "prediction" in df.columns:
                st.bar_chart(df["prediction"].value_counts())
        with c2:
            st.markdown("**Avg failure probability by health band**")
            if "health_status" in df.columns and "failure_probability" in df.columns:
                g = df.groupby("health_status")["failure_probability"].mean()
                st.bar_chart(g)
        st.markdown("**Avg failure P by prediction class**")
        if "prediction" in df.columns:
            st.bar_chart(df.groupby("prediction")["failure_probability"].mean())
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No rows logged yet.")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not load logs: {exc}")
