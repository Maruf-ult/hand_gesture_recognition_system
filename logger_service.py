# ============================================================
#  logger_service.py — Event Logging Utility
#  Hand Gesture Recognition System for Elderly People
# ============================================================

import os
import pandas as pd
from config import LOG_DIR

LOG_FILE = os.path.join(LOG_DIR, "detections.csv")

def get_logs_dataframe():
    """Load logged detections into a pandas DataFrame."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["timestamp", "gesture", "label", "priority", "confidence", "device_id"])
    try:
        return pd.read_csv(LOG_FILE)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return pd.DataFrame(columns=["timestamp", "gesture", "label", "priority", "confidence", "device_id"])

def get_log_summary():
    """Return summary analytics for detected gestures."""
    df = get_logs_dataframe()
    if df.empty:
        return {"total_alerts": 0, "critical_alerts": 0, "gesture_counts": {}}
    
    total = len(df)
    critical = len(df[df["priority"] == "CRITICAL"]) if "priority" in df.columns else 0
    counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
    return {
        "total_alerts": total,
        "critical_alerts": critical,
        "gesture_counts": counts
    }
