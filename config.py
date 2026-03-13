# ============================================================
#  config.py — Central Configuration
#  Hand Gesture Recognition System for Elderly People
# ============================================================

GESTURES = [
    "open_palm",        # HELP ME
    "closed_fist",      # PAIN
    "hand_on_chest",    # CHEST PAIN
    "pointing_down",    # NEED WASHROOM
    "hand_to_mouth",    # NEED WATER
    "fingers_to_mouth", # NEED FOOD
    "one_finger_up",    # NEED MEDICINE
    "thumbs_up",        # I AM OKAY
    "thumbs_down",      # NOT OKAY
    "hand_on_head",     # HEADACHE
    "slow_wave",        # COME HERE
    "two_fingers_up",   # NEED REST
]

GESTURE_INFO = {
    "open_palm":        {"label": "HELP ME",          "priority": "CRITICAL", "color": (0,   0,   220)},
    "closed_fist":      {"label": "PAIN",             "priority": "CRITICAL", "color": (0,   0,   200)},
    "hand_on_chest":    {"label": "CHEST PAIN",       "priority": "CRITICAL", "color": (0,   0,   180)},
    "pointing_down":    {"label": "NEED WASHROOM",    "priority": "HIGH",     "color": (0,   140, 255)},
    "hand_to_mouth":    {"label": "NEED WATER",       "priority": "HIGH",     "color": (0,   160, 255)},
    "fingers_to_mouth": {"label": "NEED FOOD",        "priority": "HIGH",     "color": (0,   180, 255)},
    "one_finger_up":    {"label": "NEED MEDICINE",    "priority": "HIGH",     "color": (200, 100, 0  )},
    "thumbs_up":        {"label": "I AM OKAY",        "priority": "LOW",      "color": (0,   200, 80 )},
    "thumbs_down":      {"label": "NOT OKAY",         "priority": "HIGH",     "color": (0,   80,  255)},
    "hand_on_head":     {"label": "HEADACHE",         "priority": "MEDIUM",   "color": (180, 0,   180)},
    "slow_wave":        {"label": "COME HERE",        "priority": "MEDIUM",   "color": (0,   180, 200)},
    "two_fingers_up":   {"label": "NEED REST",        "priority": "MEDIUM",   "color": (100, 100, 220)},
}

# Dynamic gestures (require motion)
DYNAMIC_GESTURES = ["slow_wave"]

# Gestures that always trigger urgent notification
CRITICAL_GESTURES = ["open_palm", "closed_fist", "hand_on_chest"]

# Body gestures (hand near face/head/chest — detected differently)
BODY_GESTURES = ["hand_on_chest", "hand_to_mouth", "fingers_to_mouth", "hand_on_head"]

# Data collection settings
SAMPLES_PER_GESTURE   = 500
SEQUENCE_LENGTH       = 30
DATA_DIR              = "data"
MODEL_DIR             = "model"
LOG_DIR               = "logs"

# Detection settings
CONFIDENCE_THRESHOLD  = 0.80
HOLD_SECONDS          = 1.5
SMOOTHING_BUFFER      = 10
CAMERA_WIDTH          = 1920    
CAMERA_HEIGHT         = 1080

# Notification cooldown in seconds
NOTIFICATION_COOLDOWN = 30

# Firebase
FIREBASE_CREDENTIALS  = "firebase_credentials.json"
FIREBASE_ENABLED      = False  # Set True after adding credentials