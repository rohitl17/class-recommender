"""Streamlit admin console for class recommendations."""
from __future__ import annotations

import os
from typing import List

import pandas as pd
import streamlit as st

from audit_log import log_recommendations
from data_utils import (
    PLAN_LABELS,
    client_history,
    get_subscription_plan,
    list_clients,
    load_history,
    recent_monthly_usage,
    upcoming_schedule,
)
from recommender_ai_model import (
    FrequencyRecommender,
    OpenAILLMRecommender,
    Recommendation,
    UpsellAdvisor,
)

ADMIN_USERNAME = os.environ.get("CLASS_APP_ADMIN", "admin")
ADMIN_PASSWORD = os.environ.get("CLASS_APP_PASSWORD", "fitpass123")

st.set_page_config(page_title="Class Recommender Admin", layout="wide", page_icon="💪")


@st.cache_data
def cached_history() -> pd.DataFrame:
    return load_history()


@st.cache_data
def cached_clients(history: pd.DataFrame) -> pd.DataFrame:
    return list_clients(history)


@st.cache_resource
def cached_models(
    history: pd.DataFrame,
) -> tuple[FrequencyRecommender, OpenAILLMRecommender]:
    freq_model = FrequencyRecommender(history)
    freq_model.fit()
    llm_model = OpenAILLMRecommender(fallback_model=freq_model)
    return freq_model, llm_model


@st.cache_resource
def cached_upsell_advisor() -> UpsellAdvisor:
    return UpsellAdvisor()


def ensure_session_state() -> None:
    st.session_state.setdefault("authenticated", False)


def login_panel() -> None:
    st.header("Admin access")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.success("Logged in!")
            else:
                st.error("Invalid credentials")


def render_history_table(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        st.info("No recent visits found for this client.")
        return

    metrics = history_df.groupby("class_type").size().to_dict()
    cols = st.columns(len(metrics) or 1)
    for col, (class_type, count) in zip(cols, metrics.items()):
        col.metric(class_type, count)

    st.dataframe(
        history_df[
            [
                "class_date",
                "class_day_of_week",
                "class_start_time",
                "class_type",
                "muscle_focus_upper",
                "muscle_focus_lower",
                "coach_name",
            ]
        ]
        .sort_values("class_date", ascending=False)
        .rename(
            columns={
                "class_date": "Class date",
                "class_day_of_week": "Day",
                "class_start_time": "Start time",
                "class_type": "Class style",
                "muscle_focus_upper": "Upper body focus",
                "muscle_focus_lower": "Lower body focus",
                "coach_name": "Coach",
            }
        ),
        use_container_width=True,
    )


def recommendations_to_df(recs: List[Recommendation], subscription_label: str) -> pd.DataFrame:
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "Class date": rec.class_date.strftime("%Y-%m-%d"),
                "Day": rec.class_day_of_week,
                "Start time": rec.class_start_time,
                "Class style": rec.class_type,
                "Upper focus": rec.muscle_focus_upper,
                "Lower focus": rec.muscle_focus_lower,
                "Coach": rec.coach_name,
                "Plan": subscription_label,
                "Fit score (0-1)": rec.score,
                "Why this class": rec.rationale,
            }
            for rec in recs
        ]
    )
    return df.sort_values("Class date").reset_index(drop=True)


def main() -> None:
    ensure_session_state()
    history = cached_history()
    clients = cached_clients(history)
    schedule = upcoming_schedule()
    freq_model, llm_model = cached_models(history)
    upsell_advisor = cached_upsell_advisor()

    st.title("Client Planning Console")
    st.caption("Pick a client, review their recent visits, and share the next few classes that fit their goals.")

    if not st.session_state.authenticated:
        login_panel()
        return

    with st.sidebar:
        st.subheader("Client selection")
        client_display = (clients["client_name"] + " (" + clients["client_id"] + ")").tolist()
        selected = st.selectbox(
            "Client",
            options=client_display,
            index=0 if client_display else None,
        )

        if client_display:
            selected_idx = client_display.index(selected)
            selected_row = clients.iloc[selected_idx]
            client_id = selected_row["client_id"]
            client_name = selected_row["client_name"]
        else:
            st.warning("No clients available in dataset.")
            return

    st.subheader(f"Client snapshot — {client_name}")
    history_df = client_history(history, client_id)
    subscription_plan = get_subscription_plan(history, client_id)
    subscription_label = PLAN_LABELS.get(subscription_plan, subscription_plan)
    usage = recent_monthly_usage(history, client_id)
    analysis = upsell_advisor.analyze(
        client_name,
        client_id,
        subscription_plan,
        usage,
    )
    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Current plan", subscription_label)
    if usage:
        latest = usage[0]
        metrics_cols[1].metric("Last month visits", latest["classes"])
    else:
        metrics_cols[1].metric("Last month visits", "N/A")
    metrics_cols[2].metric(
        "Upsell opportunity",
        "Yes" if analysis.get("opportunity") else "No",
    )

    st.write(
        analysis.get(
            "message",
            "Usage is aligned with the current plan.",
        )
    )

    st.subheader("Recent classes")
    render_history_table(history_df)

    st.subheader("Upcoming suggestions")
    st.caption("Both lists respect the client's preferences and show only one class per day.")
    tab_model, tab_llm = st.tabs(["Behavior patterns", "AI concierge"])

    with tab_model:
        model_recs = freq_model.recommend(client_id, schedule)
        df = recommendations_to_df(model_recs, subscription_label)
        if df.empty:
            st.info("No recommendations available.")
        else:
            st.dataframe(df, use_container_width=True)
        if st.button("Save these suggestions", key="log_freq"):
            path = log_recommendations(
                client_id, client_name, "frequency", model_recs
            )
            if path:
                st.success(f"Logged to {path}")
            else:
                st.warning("Nothing to log.")

    with tab_llm:
        llm_recs = llm_model.recommend(client_id, history_df, schedule)
        df = recommendations_to_df(llm_recs, subscription_label)
        if df.empty:
            st.info("No recommendations available.")
        else:
            st.dataframe(df, use_container_width=True)
        if st.button("Save these suggestions", key="log_llm"):
            path = log_recommendations(client_id, client_name, "llm", llm_recs)
            if path:
                st.success(f"Logged to {path}")
            else:
                st.warning("Nothing to log.")


if __name__ == "__main__":
    main()
