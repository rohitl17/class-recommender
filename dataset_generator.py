"""Utility for generating simulated historical fitness class records."""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from itertools import islice, product
from pathlib import Path
from typing import List

import pandas as pd

COACH_NAMES = [
    "Hayden Klo",
    "Avery Harve",
    "Finley Yaroslavsky",
    "Casey Hugh",
    "Devon Knight",
    "Benton Watts",
    "Tylor Tquinn",
    "Alex Blake",
    "Riley Chen",
    "Morgan Patel",
    "Sydney Romero",
    "Phoenix Green",
    "Dette Ellis",
    "Jon Flynn",
    "Kendall Jones",
]
COACHES = [(f"C{i:03}", name) for i, name in enumerate(COACH_NAMES, start=1)]

CLIENT_FIRST_NAMES = [
    "Harper",
    "Quinn",
    "Sasha",
    "Jamie",
    "Noah",
    "Cameron",
    "Emery",
    "Peyton",
    "River",
    "Logan",
]
CLIENT_LAST_NAMES = [
    "Lee",
    "Davis",
    "Wright",
    "Rivera",
    "Alvarez",
    "Diaz",
    "Doe",
    "Morgan",
    "Shah",
    "Young",
    "Hayes",
    "Coleman",
]
CLIENT_NAME_BANK = list(islice(product(CLIENT_FIRST_NAMES, CLIENT_LAST_NAMES), 100))
CLIENTS = [
    (f"CL{i:03}", f"{first} {last}")
    for i, (first, last) in enumerate(CLIENT_NAME_BANK, start=1)
]

CLASS_TYPES = ["Foundation", "Focus", "Advanced"]
UPPER_BODY_FOCUS = ["Arm Wrap", "Biceps", "Triceps", "Shoulders", "Back"]
LOWER_BODY_FOCUS = ["Leg Wrap", "Hamstrings", "Outer Glutes", "Center Glutes"]
SUBSCRIPTION_PLANS = {
    "4_per_month": 4,
    "8_per_month": 8,
    "unlimited": 16,
}
SUBSCRIPTION_CHOICES = list(SUBSCRIPTION_PLANS.keys())
DEVIATION_CHOICES = [-2, -1, 0, 0, 0, 2, 3]


def _random_date(start: date, end: date) -> date:
    if start > end:
        raise ValueError("start date must be before end date")
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def _random_time_between(start_hour: int = 5, end_hour: int = 22) -> datetime.time:
    """Return a random time rounded to the nearest 15 minutes."""
    hour = random.randint(start_hour, end_hour - 1)
    minute = random.choice((0, 15, 30, 45))
    return datetime(2000, 1, 1, hour, minute).time()


def month_iterator(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield current, min(next_month - timedelta(days=1), end)
        current = next_month


def generate_class_records(
    start_date: date | str,
    end_date: date | str,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return a DataFrame containing simulated class bookings honoring subscription plans."""
    if seed is not None:
        random.seed(seed)

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    client_subscription = {
        client_id: random.choice(SUBSCRIPTION_CHOICES) for client_id, _ in CLIENTS
    }

    records: List[dict] = []
    for client_id, client_name in CLIENTS:
        subscription = client_subscription[client_id]
        base_classes = SUBSCRIPTION_PLANS[subscription]
        for month_start, month_end in month_iterator(start_date, end_date):
            period_start = max(month_start, start_date)
            period_end = min(month_end, end_date)
            if period_start > period_end:
                continue
            deviation = random.choice(DEVIATION_CHOICES)
            target_classes = max(1, base_classes + deviation)
            available_days = (period_end - period_start).days + 1
            days = [
                period_start + timedelta(days=i) for i in range(available_days)
            ]
            if not days:
                continue
            if target_classes <= len(days):
                class_dates = random.sample(days, k=target_classes)
            else:
                class_dates = random.choices(days, k=target_classes)

            for class_date in sorted(class_dates):
                coach_id, coach_name = random.choice(COACHES)
                start_time = _random_time_between()
                class_type = random.choices(
                    CLASS_TYPES,
                    weights=(0.4, 0.35, 0.25),
                    k=1,
                )[0]
                upper_focus = random.choice(UPPER_BODY_FOCUS)
                lower_focus = random.choice(LOWER_BODY_FOCUS)

                records.append(
                    {
                        "coach_name": coach_name,
                        "coach_id": coach_id,
                        "class_start_time": start_time.strftime("%H:%M"),
                        "class_day_of_week": class_date.strftime("%A"),
                        "class_date": class_date.isoformat(),
                        "class_type": class_type,
                        "muscle_focus_upper": upper_focus,
                        "muscle_focus_lower": lower_focus,
                        "client_name": client_name,
                        "client_id": client_id,
                        "subscription_plan": subscription,
                    }
                )

    return pd.DataFrame.from_records(records)


def main() -> None:
    df = generate_class_records(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 19),
        seed=42,
    )
    output_path = Path(__file__).with_name("simulated_class_history.csv")
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
