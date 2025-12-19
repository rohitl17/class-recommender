# Class Recommender Console

A Streamlit-based planning console that helps studio admins review a client's recent fitness-class history, generate the next eight recommended sessions (both from a behavioral model and an AI concierge powered by OpenAI), and log those plans for follow-up. These recommendations can be extended to in-app recommendations for the clients to make bookings easier and reducing in-app navigation. The application also features an opportunity for upsell based on subscription usage. The project is intentionally self-contained so it can bootstrap a larger production system.

## Table of contents
1. [Purpose](#purpose)
2. [Architecture](#architecture)
3. [Key requirements](#key-requirements)
4. [Data flow](#data-flow)
5. [Recommendation engines](#recommendation-engines)
6. [Subscription/upsell insights](#subscriptionupsell-insights)
7. [Web experience](#web-experience)
8. [Running locally](#running-locally)
9. [Configuration reference](#configuration-reference)
10. [Testing](#testing)
11. [Next steps](#next-steps)
12. [Dataset generation](#dataset-generation)

## Purpose
- Give non-technical studio admins a single pane of glass to review each client's recent class activity and instantly share a short list of upcoming classes tailored to that client.
- Provide two independent recommendation pipelines: a deterministic frequency-based engine for transparency and an LLM-powered concierge for personalized narratives.
- Surface upsell/cross-sell opportunities by comparing usage against subscription tiers.
- Log every shared plan for audit/compliance purposes.

## Architecture
```
Streamlit webapp
├── data_utils.py (loading, aggregation, upcoming schedule)
├── recommender_ai_model.py
│   ├── FrequencyRecommender (behavioral)
│   ├── OpenAILLMRecommender (GPT prompt/response)
│   └── UpsellAdvisor (LLM-backed insight)
├── audit_log.py (append-only CSV logging)
├── dataset_generator.py (synthetic history + schedule inputs)
└── simulated_class_history.csv (seed data)
```
- **Data layer:** A CSV of historical class attendance is loaded into Pandas. When available, a live upcoming schedule can be injected by setting `CLASS_SCHEDULE_PATH`.
- **Model layer:** Frequency-based model relies on recency-weighted probabilities for class type, muscle focus, time bucket, and weekday. The OpenAI concierge uses richer prompts to produce JSON recommendations, enforcing "one class per day" and providing rationales.
- **Web layer:** Streamlit orchestrates login, client selection, history views, upsell messaging, and recommendation export.

## Key requirements
1. **Scalability:** Each browser session should be independent so hundreds of admins can run the console simultaneously.
2. **Transparency:** Deterministic fallback is always available; admins can see why each class was chosen.
3. **Guardrails:** Only one recommended class per date, spacing when possible, and plan usage is visibly tracked.
4. **Auditability:** Every plan can be written to `recommendation_audit.csv` with timestamps and rationale.
5. **Configurability:** Admin credentials, API keys, and live schedules are configurable via environment variables.

## Data flow
1. `simulated_class_history.csv` (or a production export) is loaded with parsed dates/times and subscription metadata.
2. `data_utils.client_history` filters the past 60 days for the selected client.
3. `data_utils.upcoming_schedule` provides the future class slate (simulated unless `CLASS_SCHEDULE_PATH` points to a real CSV).
4. Recommendations are generated; results plus subscription context flow to the UI, and optional audit log writes follow.

## Recommendation engines
### Frequency-based model
- **Inputs:** Client history, candidate schedule.
- **Scoring:** Weighted sum of preferences for class type, upper focus, lower focus, time-of-day bucket, weekday, plus freshness bonus for closer dates.
- **Recency:** Exponential decay (`exp(-days/45)`) keeps recent classes more influential.
- **Constraint:** After scoring, only one class per day survives.

### OpenAI concierge
- **Prompt content:** Client preference summary, last 15 visits, success criteria (one class per date, realistic cadence, references to history), and the candidate schedule.
- **Response:** JSON array of recommendations with date, slot, focus areas, coach, confidence, and rationale.
- **Post-processing:** Deduplicates by date and truncates to the requested count. Falls back to the frequency model when OpenAI is unavailable.

## Subscription/upsell insights
- Each client belongs to `4`, `8`, or `unlimited` classes per month.
- Usage is summarized for the last three months and compared to plan limits.
- `UpsellAdvisor` calls OpenAI (or a heuristic fallback) to determine whether to suggest a higher plan and explains why.

## Web experience
- **Login:** Basic username/password check via `CLASS_APP_ADMIN` and `CLASS_APP_PASSWORD`.
- **Client dropdown:** Populated from `client_name`/`client_id` combos.
- **Snapshot:** Shows plan, last-month visits, upsell flag, and explanation.
- **Recent classes table:** Plain-language headings and chronological sorting.
- **Recommendation tabs:** “Behavior patterns” (frequency model) and “AI concierge” (OpenAI). Each row lists class details, a plan label, fit score, and a natural-language reason.
- **Actions:** Buttons to save either set into the audit log.

## Running locally
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   (requirements: `streamlit`, `pandas`, `numpy`, `openai`)
2. **Set environment variables** (can be stored in `.env` and loaded via `os.environ`).
   ```bash
   export CLASS_APP_ADMIN="admin"
   export CLASS_APP_PASSWORD="fitpass123"
   export OPENAI_API_KEY="sk-..."   # optional but recommended
   export CLASS_SCHEDULE_PATH="/path/to/upcoming_schedule.csv"  # optional
   ```
3. **Run the console**
   ```bash
   streamlit run webapp.py
   ```
4. **Generate fresh data (optional)**
   ```bash
   python dataset_generator.py
   ```

## Configuration reference
| Variable | Description | Default |
| --- | --- | --- |
| `CLASS_APP_ADMIN` | Admin username | `admin` |
| `CLASS_APP_PASSWORD` | Admin password | `fitpass123` |
| `OPENAI_API_KEY` | OpenAI API key for recommendation + upsell LLMs | unset |
| `CLASS_SCHEDULE_PATH` | Path to CSV of upcoming classes (if omitted, simulated schedule is used) | unset |

## Testing
- Linting/formatting not yet enforced; run `python -m py_compile *.py` for syntax validation.
- Manual testing: `streamlit run webapp.py`, inspect tables, verify audit log entries, and ensure OpenAI fallbacks behave when the key is missing.

## Next steps
1. Replace the frequency model with a LightFM or implicit-ALS hybrid once real interaction data is available.
2. Store audit logs in a database with user attribution and downstream CRM automation.
3. Add multi-admin support (user accounts, activity feed).
4. Integrate email/text sharing of the recommendation cards.

## Dataset generation
Synthetic data is bootstrapped via `dataset_generator.py`. See [`DATASET_NOTES.md`](DATASET_NOTES.md) for details about how classes, subscriptions, and usage targets are randomized.
