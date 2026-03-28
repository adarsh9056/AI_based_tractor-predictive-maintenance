"""Log predictions to SQLite (default) or PostgreSQL via DATABASE_URL."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tractor_logs.db")


class Base(DeclarativeBase):
    pass


class TractorLog(Base):
    __tablename__ = "tractor_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    air_temp: Mapped[float] = mapped_column(Float)
    process_temp: Mapped[float] = mapped_column(Float)
    rpm: Mapped[float] = mapped_column(Float)
    torque: Mapped[float] = mapped_column(Float)
    tool_wear: Mapped[float] = mapped_column(Float)
    prediction: Mapped[str] = mapped_column(String(120))
    health_status: Mapped[str] = mapped_column(String(32))
    failure_probability: Mapped[float] = mapped_column(Float)
    llm_advice: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), default=lambda: datetime.now(timezone.utc).isoformat())


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def log_prediction(
    *,
    air_temp: float,
    process_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    prediction: str,
    health_status: str,
    failure_probability: float,
    llm_advice: Optional[str] = None,
) -> int:
    init_db()
    with SessionLocal() as session:
        row = TractorLog(
            air_temp=air_temp,
            process_temp=process_temp,
            rpm=rpm,
            torque=torque,
            tool_wear=tool_wear,
            prediction=prediction,
            health_status=health_status,
            failure_probability=failure_probability,
            llm_advice=llm_advice,
        )
        session.add(row)
        session.commit()
        rid = row.id
    return int(rid)


def recent_logs(
    limit: int = 50,
    offset: int = 0,
    prediction_contains: Optional[str] = None,
    health_status: Optional[str] = None,
) -> tuple[list[dict[str, Any]], int]:
    init_db()
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with SessionLocal() as session:
        q = session.query(TractorLog)
        if prediction_contains:
            q = q.filter(TractorLog.prediction.contains(prediction_contains))
        if health_status:
            q = q.filter(TractorLog.health_status == health_status)
        total = q.count()
        rows = q.order_by(TractorLog.id.desc()).offset(offset).limit(limit).all()
        items = [
            {
                "id": r.id,
                "air_temp": r.air_temp,
                "process_temp": r.process_temp,
                "rpm": r.rpm,
                "torque": r.torque,
                "tool_wear": r.tool_wear,
                "prediction": r.prediction,
                "health_status": r.health_status,
                "failure_probability": r.failure_probability,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    return items, total
