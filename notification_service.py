# ============================================================
#  notification_service.py — Notification + Logging Service
#  Hand Gesture Recognition System for Elderly People
# ============================================================

import time, csv, os
from datetime import datetime
from config import (GESTURE_INFO, CRITICAL_GESTURES,
                    NOTIFICATION_COOLDOWN, FIREBASE_ENABLED,
                    FIREBASE_CREDENTIALS, LOG_DIR)

os.makedirs(LOG_DIR, exist_ok=True)

firebase_app = None
if FIREBASE_ENABLED:
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging
        cred         = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_app = firebase_admin.initialize_app(cred)
        print("✅ Firebase connected")
    except Exception as e:
        print(f"⚠️  Firebase error: {e}")

last_sent = {}


def send_notification(gesture_name: str, confidence: float,
                      device_id: str = "laptop_cam"):
    now      = time.time()
    info     = GESTURE_INFO.get(gesture_name, {})
    label    = info.get("label", gesture_name)
    priority = info.get("priority", "MEDIUM")

    # Cooldown check
    if now - last_sent.get(gesture_name, 0) < NOTIFICATION_COOLDOWN:
        remaining = int(NOTIFICATION_COOLDOWN - (now - last_sent.get(gesture_name, 0)))
        print(f"  [Cooldown] {gesture_name} — {remaining}s left")
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Log to CSV ────────────────────────────────────────
    log_path     = os.path.join(LOG_DIR, "detections.csv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "gesture", "label",
                             "priority", "confidence", "device_id"])
        writer.writerow([timestamp, gesture_name, label,
                         priority, f"{confidence:.3f}", device_id])

    # ── Console alert ─────────────────────────────────────
    icons = {"CRITICAL": "🚨", "HIGH": "🔔", "MEDIUM": "💬", "LOW": "✅"}
    icon  = icons.get(priority, "🔔")
    print(f"\n  {icon} ALERT [{priority}] — {label}")
    print(f"     Confidence : {confidence*100:.1f}%")
    print(f"     Time       : {timestamp}")
    print(f"     Device     : {device_id}")

    # ── Firebase push notification ────────────────────────
    if FIREBASE_ENABLED and firebase_app:
        try:
            from firebase_admin import messaging
            body = f"🚨 URGENT: {label}" if priority == "CRITICAL" \
                   else f"{label} — Confidence: {confidence*100:.0f}%"

            msg = messaging.Message(
                notification=messaging.Notification(
                    title=f"Elderly Alert — {priority}",
                    body=body,
                ),
                data={
                    "gesture":    gesture_name,
                    "label":      label,
                    "priority":   priority,
                    "confidence": str(round(confidence, 3)),
                    "timestamp":  timestamp,
                    "device_id":  device_id,
                },
                topic="elderly_alerts",
            )
            messaging.send(msg)
            print(f"     Firebase   : Notification sent ✅")
        except Exception as e:
            print(f"     Firebase   : Error — {e}")

    last_sent[gesture_name] = now
    return True