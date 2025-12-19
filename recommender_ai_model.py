"""Recommendation pipelines (classic model + OpenAI LLM)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_utils import (
    PLAN_LABELS,
    client_history,
    get_subscription_plan,
    load_history,
    recent_monthly_usage,
    upcoming_schedule,
)
from dataset_generator import SUBSCRIPTION_PLANS

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - OpenAI optional
    OpenAI = None


@dataclass
class Recommendation:
    class_date: pd.Timestamp
    class_day_of_week: str
    class_start_time: str
    class_type: str
    muscle_focus_upper: str
    muscle_focus_lower: str
    coach_name: str
    coach_id: str
    score: float
    rationale: str


class FrequencyRecommender:
    """Behavioral recommender with recency weighting and cadence constraints."""

    DECAY_DAYS = 45

    def __init__(self, history_df: pd.DataFrame):
        self.history = history_df.copy()
        self._augment_history()
        self.type_pref: Dict[str, Dict[str, float]] = {}
        self.upper_pref: Dict[str, Dict[str, float]] = {}
        self.lower_pref: Dict[str, Dict[str, float]] = {}
        self.time_pref: Dict[str, Dict[str, float]] = {}
        self.weekday_pref: Dict[str, Dict[str, float]] = {}
        self.global_type: Dict[str, float] = {}
        self.global_upper: Dict[str, float] = {}
        self.global_lower: Dict[str, float] = {}
        self.global_time: Dict[str, float] = {}
        self.global_weekday: Dict[str, float] = {}

    def _augment_history(self) -> None:
        self.history["time_bucket"] = self.history["class_start_time"].apply(
            self._time_bucket
        )
        self.history["weekday"] = self.history["class_day_of_week"]
        days_since = (pd.Timestamp.today().normalize() - self.history["class_date"]).dt.days
        days_since = days_since.clip(lower=0)
        self.history["decay_weight"] = np.exp(-days_since / self.DECAY_DAYS)

    def fit(self) -> None:
        self.type_pref, self.global_type = self._build_pref("class_type")
        self.upper_pref, self.global_upper = self._build_pref("muscle_focus_upper")
        self.lower_pref, self.global_lower = self._build_pref("muscle_focus_lower")
        self.time_pref, self.global_time = self._build_pref("time_bucket")
        self.weekday_pref, self.global_weekday = self._build_pref("weekday")

    def _build_pref(
        self,
        column: str,
    ) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        grouped = (
            self.history.groupby(["client_id", column])["decay_weight"]
            .sum()
            .reset_index(name="weight")
        )
        preferences: Dict[str, Dict[str, float]] = {}
        for client_id, group in grouped.groupby("client_id"):
            total = group["weight"].sum()
            preferences[client_id] = {
                row[column]: row["weight"] / total for _, row in group.iterrows()
            }

        global_counts = (
            self.history.groupby(column)["decay_weight"].sum().reset_index(name="weight")
        )
        global_pref = {
            row[column]: row["weight"] / global_counts["weight"].sum()
            for _, row in global_counts.iterrows()
        }

        return preferences, global_pref

    def _lookup(
        self,
        pref: Dict[str, Dict[str, float]],
        client_id: str,
        key: str,
        global_pref: Dict[str, float],
    ) -> float:
        client_pref = pref.get(client_id, {})
        if key in client_pref:
            return client_pref[key]
        return global_pref.get(key, 0.0)

    def recommend(
        self,
        client_id: str,
        candidates: pd.DataFrame,
        k: int = 8,
    ) -> List[Recommendation]:
        candidates = self._augment_candidates(candidates)
        scored_rows = []
        for _, row in candidates.iterrows():
            type_score = self._lookup(
                self.type_pref, client_id, row["class_type"], self.global_type
            )
            upper_score = self._lookup(
                self.upper_pref, client_id, row["muscle_focus_upper"], self.global_upper
            )
            lower_score = self._lookup(
                self.lower_pref, client_id, row["muscle_focus_lower"], self.global_lower
            )
            time_score = self._lookup(
                self.time_pref, client_id, row["time_bucket"], self.global_time
            )
            weekday_score = self._lookup(
                self.weekday_pref, client_id, row["class_day_of_week"], self.global_weekday
            )
            days_out = max(
                0,
                (row["class_date"] - pd.Timestamp.today().normalize()).days,
            )
            freshness = max(0.0, 1.0 - days_out / 30) * 0.1
            score = (
                0.35 * type_score
                + 0.25 * upper_score
                + 0.15 * lower_score
                + 0.15 * time_score
                + 0.1 * weekday_score
                + freshness
            )
            rationale = self._build_reason(
                row,
                type_score,
                upper_score,
                lower_score,
                time_score,
                weekday_score,
            )
            scored_rows.append((score, row, rationale))

        scored_rows.sort(
            key=lambda item: (-item[0], item[1]["class_date"], item[1]["class_start_time"])
        )
        recommendations: List[Recommendation] = []
        seen_dates = set()
        for score, row, rationale in scored_rows:
            date_key = row["class_date"].date()
            if date_key in seen_dates:
                continue
            seen_dates.add(date_key)
            recommendations.append(
                Recommendation(
                    class_date=row["class_date"],
                    class_day_of_week=row["class_day_of_week"],
                    class_start_time=row["class_start_time"].strftime("%H:%M"),
                    class_type=row["class_type"],
                    muscle_focus_upper=row["muscle_focus_upper"],
                    muscle_focus_lower=row["muscle_focus_lower"],
                    coach_name=row["coach_name"],
                    coach_id=row["coach_id"],
                    score=round(score, 3),
                    rationale=rationale,
                )
            )
            if len(recommendations) >= k:
                break
        return recommendations

    def _augment_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "time_bucket" not in df.columns:
            df["time_bucket"] = df["class_start_time"].apply(self._time_bucket)
        return df

    @staticmethod
    def _time_bucket(value) -> str:
        hour = value.hour if hasattr(value, "hour") else pd.to_datetime(value).hour
        if hour < 9:
            return "Sunrise"
        if hour < 12:
            return "Morning"
        if hour < 16:
            return "Midday"
        if hour < 19:
            return "Evening"
        return "Night"

    @staticmethod
    def _build_reason(
        row: pd.Series,
        type_score: float,
        upper_score: float,
        lower_score: float,
        time_score: float,
        weekday_score: float,
    ) -> str:
        parts = []
        if type_score > 0:
            parts.append(f"Often chooses {row['class_type']} ({type_score:.2f})")
        if upper_score > 0:
            parts.append(
                f"Upper focus {row['muscle_focus_upper']} ({upper_score:.2f})"
            )
        if lower_score > 0:
            parts.append(
                f"Lower focus {row['muscle_focus_lower']} ({lower_score:.2f})"
            )
        if time_score > 0:
            parts.append(f"{row['time_bucket']} slots ({time_score:.2f})")
        if weekday_score > 0:
            parts.append(
                f"{row['class_day_of_week']} sessions ({weekday_score:.2f})"
            )
        if not parts:
            parts.append("Matches schedule availability")
        return "; ".join(parts)


class OpenAILLMRecommender:
    """Calls OpenAI to generate narrative recommendations."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        fallback_model: Optional[FrequencyRecommender] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.fallback_model = fallback_model
        self.client = None
        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def recommend(
        self,
        client_id: str,
        history_df: pd.DataFrame,
        candidates: pd.DataFrame,
        k: int = 8,
    ) -> List[Recommendation]:
        history_payload = self._prepare_history(history_df)
        upcoming_payload = self._prepare_candidates(candidates)

        if not self.client:
            return self._fallback(client_id, candidates, k)

        prompt = self._build_prompt(history_payload, upcoming_payload, k)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fitness class concierge. "
                            "Recommend upcoming classes in JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            message = response.choices[0].message.content
            parsed = json.loads(message)
            filtered = self._filter_unique_dates(parsed.get("recommendations", []), k)
            return [
                Recommendation(
                    class_date=pd.Timestamp(rec["class_date"]),
                    class_day_of_week=rec["class_day_of_week"],
                    class_start_time=rec["class_start_time"],
                    class_type=rec["class_type"],
                    muscle_focus_upper=rec["muscle_focus_upper"],
                    muscle_focus_lower=rec["muscle_focus_lower"],
                    coach_name=rec["coach_name"],
                    coach_id=rec["coach_id"],
                    score=rec.get("confidence", 0.0),
                    rationale=rec.get("rationale", "LLM suggestion"),
                )
                for rec in filtered
            ]
        except Exception as exc:  # pragma: no cover - defensive
            print(f"OpenAI request failed ({exc}); falling back to heuristic model.")
            return self._fallback(client_id, candidates, k)

    def _prepare_history(self, df: pd.DataFrame) -> Dict:
        trimmed = df.sort_values("class_date", ascending=False).head(15)
        records = []
        for _, row in trimmed.iterrows():
            records.append(
                {
                    "class_date": row["class_date"].strftime("%Y-%m-%d"),
                    "class_type": row["class_type"],
                    "muscle_focus_upper": row["muscle_focus_upper"],
                    "muscle_focus_lower": row["muscle_focus_lower"],
                    "coach_name": row["coach_name"],
                    "start_time": row["class_start_time"].strftime("%H:%M"),
                    "day": row["class_day_of_week"],
                }
            )

        summary = {
            "top_class_types": df["class_type"].value_counts().head(3).to_dict(),
            "top_upper_focus": df["muscle_focus_upper"].value_counts().head(3).to_dict(),
            "top_lower_focus": df["muscle_focus_lower"].value_counts().head(3).to_dict(),
            "preferred_days": df["class_day_of_week"].value_counts().head(3).to_dict(),
        }

        return {"recent_classes": records, "summary": summary}

    def _prepare_candidates(self, df: pd.DataFrame) -> List[Dict]:
        subset = df.head(25)
        payload = []
        for _, row in subset.iterrows():
            payload.append(
                {
                    "class_date": row["class_date"].strftime("%Y-%m-%d"),
                    "class_day_of_week": row["class_day_of_week"],
                    "class_start_time": row["class_start_time"].strftime("%H:%M"),
                    "class_type": row["class_type"],
                    "muscle_focus_upper": row["muscle_focus_upper"],
                    "muscle_focus_lower": row["muscle_focus_lower"],
                    "coach_name": row["coach_name"],
                    "coach_id": row["coach_id"],
                }
            )
        return payload

    def _build_prompt(
        self,
        history_payload: List[Dict],
        upcoming_payload: List[Dict],
        k: int,
    ) -> str:
        payload = {
            "task": {
                "summary": (
                    f"Recommend {k} upcoming classes for this client based on history and availability."
                ),
                "success_criteria": [
                    "Return at most one class per calendar date (no duplicates).",
                    "Prioritize class types and muscle focuses the client uses most.",
                    "Maintain realistic cadence: ideally ≥1 day between sessions.",
                    "Explain each suggestion in one or two sentences referencing history.",
                ],
            },
            "client_profile": history_payload["summary"],
            "recent_classes": history_payload["recent_classes"],
            "upcoming_classes": upcoming_payload,
            "instructions": [
                "Only choose sessions from the `upcoming_classes` list.",
                "If multiple classes occur on the same date, pick the best-fit slot and skip the rest.",
                "If you cannot satisfy every rule, output fewer recommendations and explain why.",
            ],
            "response_format": {
                "recommendations": [
                    {
                        "class_date": "YYYY-MM-DD",
                        "class_day_of_week": "Monday–Sunday",
                        "class_start_time": "HH:MM (24h)",
                        "class_type": "Foundation | Focus | Advanced",
                        "muscle_focus_upper": "Upper focus label",
                        "muscle_focus_lower": "Lower focus label",
                        "coach_name": "Coach name",
                        "coach_id": "Coach ID",
                        "confidence": "0-1 float expressing fit confidence",
                        "rationale": "1-2 sentence explanation referencing client preferences",
                    }
                ]
            },
        }
        return json.dumps(payload, indent=2)

    def _filter_unique_dates(self, recs: List[Dict], k: int) -> List[Dict]:
        filtered: List[Dict] = []
        seen = set()
        for rec in recs:
            date_key = rec.get("class_date")
            if not date_key or date_key in seen:
                continue
            seen.add(date_key)
            filtered.append(rec)
            if len(filtered) >= k:
                break
        return filtered

    def _fallback(
        self,
        client_id: str,
        candidates: pd.DataFrame,
        k: int,
    ) -> List[Recommendation]:
        if self.fallback_model:
            return self.fallback_model.recommend(client_id, candidates, k)
        candidates = candidates.sort_values(
            ["class_date", "class_start_time"]
        ).reset_index(drop=True)
        results = []
        seen_dates = set()
        for _, row in candidates.iterrows():
            date_key = row["class_date"].date()
            if date_key in seen_dates:
                continue
            seen_dates.add(date_key)
            results.append(
                Recommendation(
                    class_date=row["class_date"],
                    class_day_of_week=row["class_day_of_week"],
                    class_start_time=row["class_start_time"].strftime("%H:%M"),
                    class_type=row["class_type"],
                    muscle_focus_upper=row["muscle_focus_upper"],
                    muscle_focus_lower=row["muscle_focus_lower"],
                    coach_name=row["coach_name"],
                    coach_id=row["coach_id"],
                    score=0.0,
                    rationale="Fallback (soonest sessions)",
                )
            )
            if len(results) >= k:
                break
        return results


def initialize_models() -> tuple[FrequencyRecommender, OpenAILLMRecommender]:
    """Convenience helper to build both recommenders."""
    history = load_history()
    freq_model = FrequencyRecommender(history)
    freq_model.fit()
    llm_model = OpenAILLMRecommender(fallback_model=freq_model)
    return freq_model, llm_model


def preview_recommendations(client_id: str) -> Dict[str, List[Recommendation]]:
    """Utility for quick manual testing."""
    history = load_history()
    client_hist = client_history(history, client_id)
    schedule = upcoming_schedule()
    freq_model, llm_model = initialize_models()
    return {
        "frequency": freq_model.recommend(client_id, schedule),
        "llm": llm_model.recommend(client_id, client_hist, schedule),
    }


class UpsellAdvisor:
    """Determines cross-sell / upsell opportunities using an LLM."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def analyze(
        self,
        client_name: str,
        client_id: str,
        subscription_plan: str,
        monthly_usage: List[Dict],
    ) -> Dict:
        limit = SUBSCRIPTION_PLANS.get(subscription_plan, 8)
        if not monthly_usage:
            return {
                "opportunity": False,
                "message": "No recent usage data to evaluate cross-sell opportunity.",
            }

        payload = {
            "task": "Evaluate whether to upsell or cross-sell the client to a higher plan.",
            "client": {"name": client_name, "id": client_id},
            "current_plan": {
                "key": subscription_plan,
                "label": PLAN_LABELS.get(subscription_plan, subscription_plan),
                "monthly_limit": limit,
            },
            "recent_monthly_usage": monthly_usage,
            "instructions": [
                "Return JSON with fields 'opportunity' (true/false) and 'message'.",
                "Consider if the client consistently hits or exceeds their limit.",
                "Recommend only when higher engagement suggests more value.",
            ],
        }

        if not self.client:
            return self._fallback(limit, monthly_usage)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You assess fitness subscription upsell opportunities."},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as exc:
            print(f"Upsell advisor failed ({exc}); using heuristic.")
            return self._fallback(limit, monthly_usage)

    def _fallback(self, limit: int, monthly_usage: List[Dict]) -> Dict:
        latest = monthly_usage[0]["classes"]
        if latest >= limit + 1:
            return {
                "opportunity": True,
                "message": (
                    "Client is consistently using more than their plan allows. "
                    "Consider offering a higher tier."
                ),
            }
        if latest <= max(1, limit - 2):
            return {
                "opportunity": False,
                "message": "Usage is below plan limits; focus on engagement instead of upsell.",
            }
        return {
            "opportunity": False,
            "message": "Usage is aligned with the current plan.",
        }
