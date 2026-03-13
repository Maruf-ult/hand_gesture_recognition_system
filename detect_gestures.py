# ============================================================
#  detect_gestures.py
#  Hand Gesture Recognition System for Elderly People
# ============================================================

import cv2, numpy as np, pickle, time
from collections import deque, Counter
from cvzone.HandTrackingModule import HandDetector
from notification_service import send_notification
from config import (GESTURES, GESTURE_INFO, CONFIDENCE_THRESHOLD,
                    HOLD_SECONDS, SMOOTHING_BUFFER, CAMERA_WIDTH, CAMERA_HEIGHT)

with open("model/gesture_model.pkl","rb") as f: md=pickle.load(f)
model=md["model"]; scaler=md["scaler"]; GESTURES=md["gestures"]
print(f"✅ Model: {md['model_name']} | Accuracy: {md['accuracy']*100:.1f}%")

detector = HandDetector(staticMode=False, maxHands=1,
                        modelComplexity=1, detectionCon=0.7, minTrackCon=0.6)

def extract_features(lm_list):
    coords=np.array([[lm[0],lm[1],lm[2]] for lm in lm_list],dtype=float)
    coords[:,0]/=1280; coords[:,1]/=720
    wrist=coords[0]; norm=coords-wrist
    dists=np.linalg.norm(norm[[4,8,12,16,20]],axis=1)
    def angle(a,b,c):
        ba=a-b; bc=c-b
        cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
        return np.degrees(np.arccos(np.clip(cos,-1,1)))
    triplets=[(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),
              (0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),
              (14,15,16),(0,17,18),(17,18,19),(18,19,20)]
    angles=np.array([angle(coords[a],coords[b],coords[c]) for a,b,c in triplets])
    return np.concatenate([norm.flatten(),dists,angles])

PRIORITY_COLORS={"CRITICAL":(0,0,220),"HIGH":(0,140,255),
                 "MEDIUM":(0,180,100),"LOW":(100,180,100)}

pred_buffer=deque(maxlen=SMOOTHING_BUFFER)
current_gesture=""; gesture_start=0
last_confirmed=""; last_confirm_time=0
fps_buffer=deque(maxlen=30); total=0

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAMERA_HEIGHT)
print("🟢 Detection running. Press Q to quit.\n")

while True:
    t0=time.time()
    ret,frame=cap.read()
    if not ret: break
    frame=cv2.flip(frame,1)
    hands,frame=detector.findHands(frame,draw=True,flipType=False)

    raw=""; conf=0.0
    if hands:
        lm=hands[0]["lmList"]
        feat=scaler.transform([extract_features(lm)])
        proba=model.predict_proba(feat)[0]
        pred=np.argmax(proba); conf=proba[pred]
        if conf>=CONFIDENCE_THRESHOLD: raw=GESTURES[pred]

    pred_buffer.append(raw)
    if len(pred_buffer)==SMOOTHING_BUFFER:
        top_g,top_n=Counter(pred_buffer).most_common(1)[0]
        smoothed=top_g if top_n>=SMOOTHING_BUFFER*0.6 else ""
    else: smoothed=raw

    now=time.time()
    if smoothed!=current_gesture: current_gesture=smoothed; gesture_start=now
    held=now-gesture_start if current_gesture else 0
    confirmed=held>=HOLD_SECONDS and current_gesture!=""

    if confirmed and current_gesture!=last_confirmed:
        last_confirmed=current_gesture; last_confirm_time=now
        total+=1; send_notification(current_gesture,conf)
    if now-last_confirm_time>3.0: last_confirmed=""

    fps_buffer.append(1.0/(time.time()-t0+1e-6))
    fps=int(np.mean(fps_buffer))

    h,w=frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w,70),(15,15,15),-1)
    cv2.putText(frame,"Elderly Gesture Recognition System",
                (20,44),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,200,255),2)
    cv2.putText(frame,f"FPS:{fps}",(w-120,44),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,100),2)
    cv2.rectangle(frame,(0,h-140),(w,h),(15,15,15),-1)

    if current_gesture:
        info=GESTURE_INFO.get(current_gesture,{})
        label=info.get("label",current_gesture)
        priority=info.get("priority","MEDIUM")
        color=PRIORITY_COLORS.get(priority,(255,255,255))
        prefix="✔ CONFIRMED: " if confirmed else "Detecting: "
        cv2.putText(frame,prefix+label,(20,h-90),cv2.FONT_HERSHEY_SIMPLEX,
                    1.5 if confirmed else 1.0,color,3 if confirmed else 2)
        cv2.rectangle(frame,(w-160,h-140),(w,h-100),PRIORITY_COLORS.get(priority,(100,100,100)),-1)
        cv2.putText(frame,priority,(w-150,h-112),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
        bw=int(conf*380)
        cv2.rectangle(frame,(20,h-75),(400,h-52),(50,50,50),-1)
        cv2.rectangle(frame,(20,h-75),(20+bw,h-52),color,-1)
        cv2.putText(frame,f"Confidence: {conf*100:.1f}%",(410,h-54),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)
        hw=int(min(held/HOLD_SECONDS,1.0)*380)
        cv2.rectangle(frame,(20,h-44),(400,h-24),(40,40,40),-1)
        cv2.rectangle(frame,(20,h-44),(20+hw,h-24),(0,255,0) if confirmed else (0,200,255),-1)
        cv2.putText(frame,"Hold to confirm",(410,h-26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(130,130,130),1)

    cv2.putText(frame,f"Total alerts: {total}",(w-220,h-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(120,120,120),1)
    cv2.imshow("Elderly Gesture Recognition",frame)
    if cv2.waitKey(1)==ord('q'): break

cap.release(); cv2.destroyAllWindows()
print(f"\n✅ Session ended | Alerts: {total} | Log → logs/detections.csv")