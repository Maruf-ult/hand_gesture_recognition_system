# ============================================================
#  collect_data.py
#  Hand Gesture Recognition System for Elderly People
#  Python 3.13 + CVZone (no MediaPipe Tasks)
# ============================================================

import cv2, os, numpy as np, json
from cvzone.HandTrackingModule import HandDetector
from config import (GESTURES, GESTURE_INFO, DYNAMIC_GESTURES,
                    BODY_GESTURES, SAMPLES_PER_GESTURE,
                    SEQUENCE_LENGTH, DATA_DIR)

detector = HandDetector(staticMode=False, maxHands=1,
                        modelComplexity=1, detectionCon=0.6, minTrackCon=0.5)

for g in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)
    if g in DYNAMIC_GESTURES:
        os.makedirs(os.path.join(DATA_DIR, g + "_seq"), exist_ok=True)
os.makedirs("logs", exist_ok=True)

def extract_features(lm_list, hand_type="Right"):
    coords = np.array([[lm[0], lm[1], lm[2]] for lm in lm_list], dtype=float)
    coords[:, 0] /= 1280.0
    coords[:, 1] /= 720.0
    wrist = coords[0].copy()
    norm  = coords - wrist
    wrist_x = wrist[0]
    wrist_y = wrist[1]
    
    # Anatomical handedness check (cross product of Wrist->Middle and Wrist->Pinky)
    # Independent of frame flipping or MediaPipe hand_type label!
    v_mid = norm[9]
    v_pnk = norm[17]
    cp = v_mid[0] * v_pnk[1] - v_mid[1] * v_pnk[0]
    if cp < 0:
        norm[:, 0] = -norm[:, 0]
        wrist_x = 1.0 - wrist_x

    dists = np.linalg.norm(norm[[4,8,12,16,20]], axis=1)
    def angle(a, b, c):
        ba = a-b; bc = c-b
        cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
        return np.degrees(np.arccos(np.clip(cos,-1.0,1.0)))
    triplets = [(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),
                (0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),
                (14,15,16),(0,17,18),(17,18,19),(18,19,20)]
    angles = np.array([angle(coords[a],coords[b],coords[c]) for a,b,c in triplets])
    wrist_pos = np.array([wrist_x, wrist_y])
    return np.concatenate([norm.flatten(), dists, angles, wrist_pos])



def get_hand_crop(frame, lm_list):
    h, w = frame.shape[:2]
    xs = [lm[0] for lm in lm_list]; ys = [lm[1] for lm in lm_list]
    x1=max(0,min(xs)-30); x2=min(w,max(xs)+30)
    y1=max(0,min(ys)-30); y2=min(h,max(ys)+30)
    return frame[y1:y2, x1:x2]

INSTRUCTIONS = {
    "open_palm":        "Raise open hand — all fingers spread wide",
    "closed_fist":      "Gently close hand into a fist and raise it",
    "hand_on_chest":    "Place open hand flat on your chest",
    "pointing_down":    "Make fist — point index finger downward",
    "hand_to_mouth":    "Cup hand and bring toward your mouth",
    "fingers_to_mouth": "Pinch fingers and bring to mouth (eating)",
    "one_finger_up":    "Make fist — raise only index finger upward",
    "thumbs_up":        "Make fist — stick thumb straight up",
    "thumbs_down":      "Make fist — point thumb straight down",
    "hand_on_head":     "Place open hand flat on top of your head",
    "slow_wave":        "Raise open hand — hold steadily with fingers together",

    "two_fingers_up":   "Make fist — raise index + middle finger together",
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
collection_log = {}

print("\n✅ CVZone loaded\n🟢 Collection Started\nSPACE=Start | S=Skip | Q=Quit")

for gesture in GESTURES:
    info = GESTURE_INFO[gesture]
    is_dynamic = gesture in DYNAMIC_GESTURES
    is_body    = gesture in BODY_GESTURES
    print(f"\n>>> {gesture.upper()} | {info['label']} | {INSTRUCTIONS[gesture]}")

    skipped = False
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w,110), (15,15,15), -1)
        cv2.putText(frame, f"NEXT: {gesture.upper().replace('_',' ')}",
                    (24,48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,200,255), 2)
        cv2.putText(frame, f"Means: {info['label']}",
                    (24,82), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180,180,180), 1)
        cv2.rectangle(frame, (0,115), (w,162), (30,30,30), -1)
        cv2.putText(frame, f"HOW: {INSTRUCTIONS[gesture]}",
                    (24,146), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,220,100), 2)
        cv2.rectangle(frame, (0,h-45), (w,h), (15,15,15), -1)
        cv2.putText(frame, "SPACE=Start  |  S=Skip  |  Q=Quit",
                    (24,h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150,150,150), 1)
        cv2.imshow("Gesture Collection", frame)
        key = cv2.waitKey(1)
        if key == ord(' '): break
        if key == ord('s'): skipped=True; break
        if key == ord('q'): cap.release(); cv2.destroyAllWindows(); exit()
    if skipped: continue

    count=0; save_path=os.path.join(DATA_DIR,gesture); features_list=[]

    if is_dynamic:
        seqs=0; target=SAMPLES_PER_GESTURE//SEQUENCE_LENGTH
        while seqs < target:
            sequence=[]
            while len(sequence) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                hands, frame = detector.findHands(frame, draw=True, flipType=False)
                if hands:
                    hand_type = hands[0].get("type", "Right")
                    sequence.append(extract_features(hands[0]["lmList"], hand_type).tolist())
                h,w=frame.shape[:2]; bar=int(len(sequence)/SEQUENCE_LENGTH*400)
                cv2.rectangle(frame,(0,0),(w,60),(15,15,15),-1)
                cv2.putText(frame,f"{gesture.upper().replace('_',' ')} Seq {seqs+1}/{target}",
                            (24,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,255),2)
                cv2.rectangle(frame,(24,h-50),(424,h-24),(50,50,50),-1)
                cv2.rectangle(frame,(24,h-50),(24+bar,h-24),(0,220,100),-1)
                cv2.imshow("Gesture Collection", frame); cv2.waitKey(1)
            with open(os.path.join(DATA_DIR,gesture+"_seq",f"seq_{seqs}.json"),"w") as f:
                json.dump(sequence,f)
            seqs+=1
        collection_log[gesture]={"type":"dynamic","sequences":seqs}
    else:
        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            hands, frame = detector.findHands(frame, draw=True, flipType=False)
            h,w = frame.shape[:2]
            if hands:
                lm = hands[0]["lmList"]
                hand_type = hands[0].get("type", "Right")
                crop = get_hand_crop(frame, lm)
                if crop.size > 0:
                    cv2.imwrite(os.path.join(save_path,f"{count}.jpg"), crop)
                features_list.append(extract_features(lm, hand_type).tolist())
                count+=1

            bar=int(count/SAMPLES_PER_GESTURE*400)
            cv2.rectangle(frame,(0,0),(w,60),(15,15,15),-1)
            cv2.putText(frame,f"{gesture.upper().replace('_',' ')} [{count}/{SAMPLES_PER_GESTURE}]",
                        (24,40),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,200,255),2)
            cv2.rectangle(frame,(24,h-50),(424,h-24),(50,50,50),-1)
            cv2.rectangle(frame,(24,h-50),(24+bar,h-24),(0,220,100),-1)
            cv2.putText(frame,f"{int(count/SAMPLES_PER_GESTURE*100)}%",
                        (440,h-26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,180,180),1)
            cv2.rectangle(frame,(0,h-90),(w,h-55),(25,25,25),-1)
            cv2.putText(frame,INSTRUCTIONS[gesture],
                        (24,h-66),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,220,100),1)
            cv2.imshow("Gesture Collection", frame)
            if cv2.waitKey(1)==ord('q'): break
        with open(os.path.join(save_path,"_features.json"),"w") as f:
            json.dump(features_list,f)
        collection_log[gesture]={"type":"static","samples":count}
        print(f"  ✅ {gesture} — {count} samples")

cap.release(); cv2.destroyAllWindows()
with open("logs/collection_log.json","w") as f: json.dump(collection_log,f,indent=2)
print("\n✅ Done! Run: python train_model.py")