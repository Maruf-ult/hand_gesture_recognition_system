# ============================================================
#  generate_dummy_data.py — Synthetic Data Generator
#  Hand Gesture Recognition System for Elderly People
# ============================================================

import os
import json
import numpy as np
from config import GESTURES, DYNAMIC_GESTURES, DATA_DIR, SAMPLES_PER_GESTURE

os.makedirs(DATA_DIR, exist_ok=True)

print("[INFO] Generating synthetic hand landmark feature data for initial model testing...")

# Feature dimensions: 21 landmarks (3D) -> 63 coords + 5 tip dists + 15 angles + 2 wrist pos = 85 features
N_FEATURES = 85
SAMPLES = 200

np.random.seed(42)

# Specific spatial heights for body gestures to distinguish hand_on_chest vs open_palm vs hand_on_head
HEIGHT_MAP = {
    "hand_on_head": 0.15,
    "open_palm": 0.35,
    "hand_to_mouth": 0.30,
    "fingers_to_mouth": 0.30,
    "hand_on_chest": 0.65,
}

for idx, gesture in enumerate(GESTURES):
    if gesture in DYNAMIC_GESTURES:
        continue
        
    folder = os.path.join(DATA_DIR, gesture)
    os.makedirs(folder, exist_ok=True)
    
    # Base pattern
    base = np.zeros(N_FEATURES)
    base[:63] = np.sin(np.linspace(0, np.pi * (idx + 1), 63)) * 0.2
    base[63:68] = 0.1 + 0.05 * (idx % 4)   # distances
    base[68:83] = 45.0 + 10.0 * (idx % 6)  # angles
    
    # Spatial wrist position
    base[83] = 0.5  # wrist_x
    base[84] = HEIGHT_MAP.get(gesture, 0.45)  # wrist_y height
    
    # Add random noise
    noise = np.random.normal(0, 0.02, size=(SAMPLES, N_FEATURES))
    noise[:, 68:83] *= 10  # scale angle noise
    
    samples = base + noise
    features_list = samples.tolist()
    
    feat_file = os.path.join(folder, "_features.json")
    with open(feat_file, "w") as f:
        json.dump(features_list, f)
        
    print(f"  [OK] {gesture:<20} -> {SAMPLES} synthetic samples saved to {feat_file}")


print("\n[SUCCESS] Synthetic dataset created successfully!")
print("Run `python train_model.py` to train your initial model.")
