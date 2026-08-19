# ============================================================
#  dashboard.py — Monitoring Dashboard (Streamlit)
#  Hand Gesture Recognition System for Elderly People
# ============================================================

import streamlit as st
import pandas as pd
import os
import time
from logger_service import get_logs_dataframe, get_log_summary
from config import GESTURE_INFO, GESTURES

st.set_page_config(
    page_title="Elderly Gesture Care Monitor",
    page_icon="👵",
    layout="wide"
)

st.title("👵 Elderly Hand Gesture Monitoring System")
st.markdown("Real-time monitoring and analytics dashboard for emergency alerts and care requests.")

# Sidebar Controls
st.sidebar.header("Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh (every 5s)", value=True)
selected_priority = st.sidebar.multiselect(
    "Filter by Priority",
    options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
)

# Fetch Data
df = get_logs_dataframe()

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

total_alerts = len(df)
critical_alerts = len(df[df["priority"] == "CRITICAL"]) if not df.empty and "priority" in df.columns else 0
high_alerts = len(df[df["priority"] == "HIGH"]) if not df.empty and "priority" in df.columns else 0
unique_gestures = df["gesture"].nunique() if not df.empty and "gesture" in df.columns else 0

col1.metric("Total Alerts", total_alerts)
col2.metric("Critical Emergency Alerts", critical_alerts, delta_color="inverse")
col3.metric("High Priority Care Requests", high_alerts)
col4.metric("Unique Gestures Detected", unique_gestures)

st.markdown("---")

if not df.empty:
    # Filter DataFrame by priority
    if "priority" in df.columns and selected_priority:
        filtered_df = df[df["priority"].isin(selected_priority)]
    else:
        filtered_df = df

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📊 Gesture Frequency")
        if "label" in filtered_df.columns and not filtered_df.empty:
            gesture_counts = filtered_df["label"].value_counts().reset_index()
            gesture_counts.columns = ["Gesture", "Count"]
            st.bar_chart(data=gesture_counts, x="Gesture", y="Count", use_container_width=True)
        else:
            st.info("No logs match the selected filter.")

    with right_col:
        st.subheader("🚨 Priority Breakdown")
        if "priority" in filtered_df.columns and not filtered_df.empty:
            priority_counts = filtered_df["priority"].value_counts().reset_index()
            priority_counts.columns = ["Priority", "Count"]
            st.dataframe(priority_counts, use_container_width=True)
        else:
            st.info("No data available.")

    st.markdown("---")
    st.subheader("📋 Detection Logs")
    st.dataframe(filtered_df.sort_index(ascending=False), use_container_width=True)

else:
    st.info("No detection logs found yet. Run `detect_gestures.py` to start monitoring.")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Gestures Configured:**")
for g in GESTURES:
    info = GESTURE_INFO.get(g, {})
    st.sidebar.text(f"• {info.get('label', g)} ({info.get('priority', 'N/A')})")

if auto_refresh:
    time.sleep(5)
    st.rerun()
