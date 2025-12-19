# Dataset generation guide

This document explains how `dataset_generator.py` creates the synthetic history stored in `simulated_class_history.csv`. It is intended for product and data stakeholders who need to understand how the sample data behaves and how to tweak it.

## Goals
- Provide a realistic (yet anonymized) attendance log for roughly 100 clients and 15 coaches.
- Encode subscription tiers so recommendation models can reason about per-month limits.
- Simulate monthly usage patterns that hover around each subscription level (with small random variations).
- Produce inputs compatible with the Streamlit console and both recommendation pipelines.

## Entity pools
- **Coaches:** 15 unique names with IDs `C001`–`C015`.
- **Clients:** 100 name combinations from curated first/last-name lists, IDs `CL001`–`CL100`.
- **Class metadata:**
  - Class styles: `Foundation`, `Focus`, `Advanced` (weighted 40/35/25).
  - Upper-body focus: `Arm Wrap`, `Biceps`, `Triceps`, `Shoulders`, `Back`.
  - Lower-body focus: `Leg Wrap`, `Hamstrings`, `Outer Glutes`, `Center Glutes`.

## Subscription plans
```python
SUBSCRIPTION_PLANS = {
    "4_per_month": 4,
    "8_per_month": 8,
    "unlimited": 16,
}
```
- Every client is randomly assigned one of the plans once, and that plan persists for the entire dataset.
- The numeric value indicates the expected number of classes per month (unlimited is capped at `16` for simulation purposes).

## Monthly generation logic
1. Iterate over each client.
2. Iterate over each month between `start_date` and `end_date`.
3. Determine the target number of classes for that month:
   - Start from the plan limit (4, 8, or 16).
   - Add a deviation randomly chosen from `[-2, -1, 0, 0, 0, +2, +3]`.
   - Clamp to a minimum of 1 class.
4. Randomly pick that many distinct days within the month (or reuse days if the target exceeds available days, which only happens for unlimited). Every scheduled day receives:
   - A coach (uniform random).
   - A start time rounded to the nearest 15 minutes between 5 a.m. and 10 p.m.
   - Class type and muscle focuses drawn from the configured distributions.
5. Record the row with the client metadata, coach metadata, class details, and `subscription_plan` label.

## Output schema
| Column | Description |
| --- | --- |
| `coach_name`, `coach_id` | Coach metadata |
| `class_start_time` | `HH:MM` string (15-minute increments) |
| `class_day_of_week`, `class_date` | Day name + ISO date string |
| `class_type` | `Foundation`, `Focus`, or `Advanced` |
| `muscle_focus_upper`, `muscle_focus_lower` | Upper/lower body focus labels |
| `client_name`, `client_id` | Client metadata |
| `subscription_plan` | `4_per_month`, `8_per_month`, or `unlimited` |

## Reproducing / tweaking
- **Regenerate the dataset:** `python dataset_generator.py`
- **Change timeframe:** edit the `start_date` / `end_date` in `main()`.
- **Adjust plan mix:** alter `SUBSCRIPTION_CHOICES` or the plan dictionary.
- **Bias class types:** tweak the weights in `random.choices(CLASS_TYPES, weights=...)`.
- **Increase granularity:** add additional focuses or class properties and update downstream code to load them.

## Notes
- Because the data is synthetic, it can be safely committed or shared publicly.
- Subscription labels feed directly into the Streamlit UI and upsell logic; any schema changes should be mirrored in `data_utils.py`.
- The generator seeds Python’s `random` module when `seed` is provided, allowing reproducible datasets for testing.

