"""Data loading and scheduling helpers for the class recommender app."""
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from dataset_generator import (
    CLASS_TYPES,
    COACHES,
    LOWER_BODY_FOCUS,
    SUBSCRIPTION_PLANS,
    UPPER_BODY_FOCUS,
)

DATA_PATH = Path(__file__).with_name("simulated_class_history.csv")
SCHEDULE_PATH = os.getenv("CLASS_SCHEDULE_PATH")
PLAN_LABELS = {
    "4_per_month": "4 classes / month",
    "8_per_month": "8 classes / month",
    "unlimited": "Unlimited",
}


def load_history(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the historical dataset and parse datetimes."""
    df = pd.read_csv(path)
    df["class_date"] = pd.to_datetime(df["class_date"])
    df["class_start_time"] = pd.to_datetime(df["class_start_time"], format="%H:%M").dt.time
    df["class_day_of_week"] = df["class_date"].dt.day_name()
    return df


def list_clients(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique client id/name pairs sorted alphabetically."""
    clients = (
        df[["client_id", "client_name"]]
        .drop_duplicates()
        .sort_values("client_name")
        .reset_index(drop=True)
    )
    return clients


def client_history(
    df: pd.DataFrame,
        client_id: str,
        window_days: int = 60,
) -> pd.DataFrame:
    """Return the most recent records for a given client."""
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=window_days)
    hist = (
        df[df["client_id"] == client_id]
        .copy()
        .sort_values("class_date", ascending=False)
    )
    return hist[hist["class_date"] >= cutoff]


def get_subscription_plan(df: pd.DataFrame, client_id: str) -> str:
    records = df[df["client_id"] == client_id]
    if records.empty:
        return "8_per_month"
    return (
        records["subscription_plan"]
        .mode()
        .iloc[0]
        if "subscription_plan" in records.columns
        else "8_per_month"
    )


def recent_monthly_usage(
    df: pd.DataFrame,
    client_id: str,
    months: int = 3,
) -> List[dict]:
    client_rows = df[df["client_id"] == client_id]
    if client_rows.empty:
        return []
    client_rows = client_rows.copy()
    client_rows["year_month"] = client_rows["class_date"].dt.to_period("M")
    usage = (
        client_rows.groupby("year_month")
        .size()
        .reset_index(name="classes")
        .sort_values("year_month", ascending=False)
        .head(months)
    )
    return [
        {"month": str(row["year_month"]), "classes": int(row["classes"])}
        for _, row in usage.iterrows()
    ]


def generate_upcoming_schedule(
    days_ahead: int = 30,
    sessions_per_day: int = 6,
    seed: int | None = 123,
) -> pd.DataFrame:
    """Simulate an upcoming class schedule."""
    rng = random.Random(seed)
    today = datetime.today().date()
    rows: List[Dict] = []

    for delta in range(days_ahead):
        class_date = today + timedelta(days=delta)
        for session in range(sessions_per_day):
            class_type = rng.choice(CLASS_TYPES)
            upper = rng.choice(UPPER_BODY_FOCUS)
            lower = rng.choice(LOWER_BODY_FOCUS)
            coach_id, coach_name = rng.choice(COACHES)
            hour = rng.choice(range(5, 22))
            minute = rng.choice((0, 15, 30, 45))
            start_time = datetime.combine(class_date, datetime.min.time()).replace(
                hour=hour,
                minute=minute,
            )

            rows.append(
                {
                    "class_date": class_date,
                    "class_day_of_week": class_date.strftime("%A"),
                    "class_start_time": start_time.time(),
                    "class_type": class_type,
                    "muscle_focus_upper": upper,
                    "muscle_focus_lower": lower,
                    "coach_id": coach_id,
                    "coach_name": coach_name,
                }
            )

    schedule = (
        pd.DataFrame(rows)
        .assign(class_date=lambda df: pd.to_datetime(df["class_date"]))
        .sort_values(["class_date", "class_start_time"])
        .reset_index(drop=True)
    )
    return schedule


def _prepare_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["class_date"] = pd.to_datetime(df["class_date"])
    if not pd.api.types.is_datetime64_any_dtype(df["class_start_time"]):
        df["class_start_time"] = pd.to_datetime(
            df["class_start_time"], format="%H:%M"
        ).dt.time
    if "class_day_of_week" not in df.columns:
        df["class_day_of_week"] = df["class_date"].dt.day_name()
    return df[
        [
            "class_date",
            "class_day_of_week",
            "class_start_time",
            "class_type",
            "muscle_focus_upper",
            "muscle_focus_lower",
            "coach_id",
            "coach_name",
        ]
    ].sort_values(["class_date", "class_start_time"])


def load_real_schedule(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _prepare_schedule(df)


def upcoming_schedule() -> pd.DataFrame:
    """Return either a real schedule (if provided) or a simulated one."""
    if SCHEDULE_PATH and Path(SCHEDULE_PATH).exists():
        return load_real_schedule(SCHEDULE_PATH)
    return generate_upcoming_schedule()
