# ============================================================
#  train_model.py
#  Hand Gesture Recognition System for Elderly People
# ============================================================

import numpy as np, os, json, pickle, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from cvzone.HandTrackingModule import HandDetector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from config import GESTURES, DYNAMIC_GESTURES, DATA_DIR, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

detector = HandDetector(staticMode=True, maxHands=1, detectionCon=0.5)

def extract_features(lm_list):
    coords = np.array([[lm[0],lm[1],lm[2]] for lm in lm_list], dtype=float)
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

print("Loading data..."); print("="*50)
data, labels = [], []

for idx, gesture in enumerate(GESTURES):
    if gesture in DYNAMIC_GESTURES:
        print(f"  ⏭  {gesture} (dynamic — skipped)"); continue
    folder = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(folder):
        print(f"  ⚠️  {gesture} — no folder, skipping"); continue

    feat_file = os.path.join(folder, "_features.json")
    if os.path.exists(feat_file):
        with open(feat_file) as f: feats=json.load(f)
        for feat in feats: data.append(feat); labels.append(idx)
        print(f"  ✅ {gesture:<22} {len(feats):>4} samples"); continue

    imgs=[f for f in os.listdir(folder) if f.endswith(".jpg")]
    count=0
    for img_file in imgs:
        img=cv2.imread(os.path.join(folder,img_file))
        if img is None: continue
        hands,_=detector.findHands(img,draw=False,flipType=False)
        if hands:
            data.append(extract_features(hands[0]["lmList"]).tolist())
            labels.append(idx); count+=1
    print(f"  ✅ {gesture:<22} {count:>4} samples")

print(f"\nTotal: {len(data)} samples | {len(set(labels))} gestures")
if len(data)==0: print("❌ No data found. Run collect_data.py first."); exit()

X=np.array(data); y=np.array(labels)
scaler=StandardScaler(); X_scaled=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,stratify=y,random_state=42)

models = {
    "Random Forest":    RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1),
    "Neural Network":   MLPClassifier(hidden_layer_sizes=(256,128,64),activation="relu",
                                      max_iter=500,early_stopping=True,random_state=42),
    "Gradient Boost":   GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,
                                                    max_depth=5,random_state=42),
}

best_model=None; best_acc=0; best_name=""; results={}
for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train,y_train)
    acc=accuracy_score(y_test,model.predict(X_test))
    cv=cross_val_score(model,X_scaled,y,cv=5)
    results[name]={"accuracy":acc,"cv_mean":cv.mean(),"cv_std":cv.std()}
    print(f"    Accuracy: {acc*100:.2f}%  |  CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    if acc>best_acc: best_acc=acc; best_model=model; best_name=name

print(f"\n🏆 Best: {best_name} — {best_acc*100:.2f}%")

gesture_names=[GESTURES[i].replace("_"," ") for i in sorted(set(y))]
y_pred=best_model.predict(X_test)
print("\n"+classification_report(y_test,y_pred,target_names=gesture_names))

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(14,10))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=gesture_names,yticklabels=gesture_names)
plt.title(f"Confusion Matrix — {best_name} ({best_acc*100:.1f}%)",fontsize=14)
plt.xticks(rotation=45,ha="right",fontsize=9); plt.yticks(fontsize=9)
plt.tight_layout(); plt.savefig("logs/confusion_matrix.png",dpi=150)
print("  Saved → logs/confusion_matrix.png")

with open(os.path.join(MODEL_DIR,"gesture_model.pkl"),"wb") as f:
    pickle.dump({"model":best_model,"scaler":scaler,"gestures":GESTURES,
                 "model_name":best_name,"accuracy":best_acc},f)
print(f"\n✅ Model saved → {MODEL_DIR}/gesture_model.pkl")
print("   Next step: python detect_gestures.py")