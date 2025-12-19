"""Simple CSV audit logging for generated recommendations."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from recommender_ai_model import Recommendation

LOG_PATH = Path(__file__).with_name("recommendation_audit.csv")


def log_recommendations(
    client_id: str,
    client_name: str,
    engine: str,
    recommendations: List[Recommendation],
) -> Path | None:
    """Append recommendation rows to the audit CSV."""
    if not recommendations:
        return None

    timestamp = datetime.utcnow().isoformat()
    rows = []
    for rec in recommendations:
        rows.append(
            {
                "timestamp": timestamp,
                "engine": engine,
                "client_id": client_id,
                "client_name": client_name,
                "class_date": rec.class_date.strftime("%Y-%m-%d"),
                "class_day_of_week": rec.class_day_of_week,
                "class_start_time": rec.class_start_time,
                "class_type": rec.class_type,
                "muscle_focus_upper": rec.muscle_focus_upper,
                "muscle_focus_lower": rec.muscle_focus_lower,
                "coach_name": rec.coach_name,
                "coach_id": rec.coach_id,
                "score": rec.score,
                "rationale": rec.rationale,
                "payload": json.dumps(
                    {
                        "class_date": rec.class_date.strftime("%Y-%m-%d"),
                        "class_day_of_week": rec.class_day_of_week,
                        "class_start_time": rec.class_start_time,
                        "class_type": rec.class_type,
                        "muscle_focus_upper": rec.muscle_focus_upper,
                        "muscle_focus_lower": rec.muscle_focus_lower,
                        "coach_name": rec.coach_name,
                        "coach_id": rec.coach_id,
                        "score": rec.score,
                        "rationale": rec.rationale,
                    }
                ),
            }
        )

    df = pd.DataFrame(rows)
    header = not LOG_PATH.exists()
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)
    return LOG_PATH
