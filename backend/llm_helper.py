"""RAG-style retrieval from tractor_fault_manual.csv + optional OpenAI diagnosis."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANUAL_CSV = ROOT / "data" / "tractor_fault_manual.csv"


def _load_rows() -> list[dict[str, str]]:
    if not MANUAL_CSV.exists():
        return []
    with MANUAL_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _score_row(prediction_label: str, r: dict[str, str]) -> int:
    key = prediction_label.lower()
    blob = f"{r.get('condition', '')} {r.get('diagnosis', '')} {r.get('action', '')}".lower()
    score = sum(1 for tok in key.replace("/", " ").split() if len(tok) > 3 and tok in blob)
    if "healthy" in key and "healthy" in blob:
        score += 3
    if "heat" in key and "heat" in blob:
        score += 2
    if "power" in key and "power" in blob:
        score += 2
    if "overstrain" in key and "overstrain" in blob.lower():
        score += 2
    if "tool wear" in key and "tool" in blob:
        score += 2
    if "random" in key and "random" in blob:
        score += 2
    return score


def retrieve_citations(prediction_label: str, max_rows: int = 4) -> list[dict[str, Any]]:
    """Return scored manual rows as structured citations (trust layer for UI / LLM)."""
    rows = _load_rows()
    if not rows:
        return []
    scored = [(_score_row(prediction_label, r), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [r for s, r in scored if s > 0][:max_rows] or rows[:max_rows]
    out: list[dict[str, Any]] = []
    for r in picked:
        out.append(
            {
                "condition": r.get("condition", ""),
                "diagnosis": r.get("diagnosis", ""),
                "action": r.get("action", ""),
                "match_score": _score_row(prediction_label, r),
            }
        )
    return out


def retrieve_context(prediction_label: str, max_rows: int = 4) -> str:
    cites = retrieve_citations(prediction_label, max_rows)
    if not cites:
        return "No maintenance manual rows loaded."
    parts = []
    for i, c in enumerate(cites, 1):
        parts.append(
            f"[Citation {i}] Condition: {c['condition']}\nDiagnosis: {c['diagnosis']}\nAction: {c['action']}"
        )
    return "\n\n".join(parts)


def diagnosis_bundle(
    *,
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    prediction: str,
    health_status: str,
    failure_probability: float,
) -> dict[str, Any]:
    citations = retrieve_citations(prediction)
    context = retrieve_context(prediction)
    api_key = os.getenv("OPENAI_API_KEY")
    text: str
    if not api_key:
        text = _offline_advice(prediction, context, health_status, failure_probability, citations)
        return {"llm_diagnosis": text, "manual_citations": citations}

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2, api_key=api_key)
        sys = SystemMessage(
            content=(
                "You are a senior agricultural machinery field engineer. "
                "Ground recommendations in the numbered citations; if a claim is not supported by a citation, say so. "
                "Respond with: Urgency, Likely cause, Recommended action, Repair priority (P1–P4), "
                "and a final line 'Citations used: [1], [2], ...' referencing only provided snippets."
            )
        )
        human = HumanMessage(
            content=(
                f"Sensors: air_temp_K={air_temp}, process_temp_K={process_temp}, rpm={rpm}, "
                f"torque_Nm={torque}, tool_wear_min={tool_wear}.\n"
                f"ML prediction: {prediction}. Health band: {health_status}. "
                f"Estimated failure probability: {failure_probability:.2%}.\n\n"
                f"Retrieved manual excerpts:\n{context}"
            )
        )
        out = llm.invoke([sys, human])
        text = str(out.content).strip()
    except Exception as exc:  # noqa: BLE001
        text = (
            _offline_advice(prediction, context, health_status, failure_probability, citations)
            + f"\n\n(LLM unavailable: {exc})"
        )
    return {"llm_diagnosis": text, "manual_citations": citations}


def _offline_advice(
    prediction: str,
    context: str,
    health_status: str,
    failure_probability: float,
    citations: list[dict[str, Any]],
) -> str:
    cite_note = (
        f"Citations ({len(citations)} manual row(s)): "
        + "; ".join(f"[{i+1}] {c.get('diagnosis', '')[:80]}" for i, c in enumerate(citations))
        if citations
        else "No manual citations."
    )
    return (
        f"Urgency: {health_status} (heuristic band; ML failure estimate {failure_probability:.1%})\n"
        f"Likely cause: {prediction}\n"
        f"Recommended action (manual retrieval):\n{context}\n"
        f"{cite_note}\n"
        "Repair priority: P1 if Critical, P2 if Warning, P4 if Healthy."
    )


def diagnosis_from_llm(
    *,
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    prediction: str,
    health_status: str,
    failure_probability: float,
) -> str:
    """Backward-compatible string return for callers that only need text."""
    return str(
        diagnosis_bundle(
            air_temp=air_temp,
            process_temp=process_temp,
            rpm=rpm,
            torque=torque,
            tool_wear=tool_wear,
            prediction=prediction,
            health_status=health_status,
            failure_probability=failure_probability,
        )["llm_diagnosis"]
    )
