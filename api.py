"""
HealthFusion AI v3 — FastAPI Backend
Includes: SQLite DB, Auth, All ML endpoints, PDF, Skin CNN
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import pickle, os, io, hashlib, sqlite3, secrets
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Optional ──
try:
    import shap; SHAP_AVAILABLE = True # Installed via pip
except: SHAP_AVAILABLE = False

try:
    import torch, torch.nn as nn
    from torchvision import models as tvm, transforms
    from PIL import Image as PILImage
    TORCH_AVAILABLE = True
except: TORCH_AVAILABLE = False

try:
    import bcrypt; BCRYPT_AVAILABLE = True
except: BCRYPT_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    REPORTLAB = True
except: REPORTLAB = False

# ─────────────────────────────────────────────
#  FASTAPI INIT
# ─────────────────────────────────────────────
app = FastAPI(title="HealthFusion AI", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "healthfusion.db"
SESSIONS = {}  # token → user_id (in-memory sessions)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# The new google-genai package uses a standard client initialization down below
# so no global configure() call is needed here.

# ─────────────────────────────────────────────
#  DATABASE SETUP
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            pred_type TEXT,
            heart_prob REAL,
            diabetes_prob REAL,
            uhri REAL,
            risk_level TEXT,
            disease_name TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()

init_db()

def hash_pw(pw: str) -> str:
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    return hashlib.sha256(pw.encode()).hexdigest()

def check_pw(pw: str, hashed: str) -> bool:
    if BCRYPT_AVAILABLE:
        try: return bcrypt.checkpw(pw.encode(), hashed.encode())
        except: return hashlib.sha256(pw.encode()).hexdigest() == hashed
    return hashlib.sha256(pw.encode()).hexdigest() == hashed

def save_prediction(user_id, pred_type, heart_prob=None, diabetes_prob=None,
                    uhri=None, risk_lv=None, disease_name=None, confidence=None):
    try:
        conn = get_db()
        conn.execute("""
            INSERT INTO predictions (user_id, pred_type, heart_prob, diabetes_prob,
                uhri, risk_level, disease_name, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, pred_type, heart_prob, diabetes_prob, uhri, risk_lv, disease_name, confidence))
        conn.commit()
        conn.close()
    except: pass

def get_history(user_id, limit=20):
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT pred_type, heart_prob, diabetes_prob, uhri, risk_level,
                   disease_name, confidence, timestamp
            FROM predictions
            WHERE user_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (user_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except: return []

def get_uhri_trend(user_id, limit=10):
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT uhri, timestamp FROM predictions
            WHERE user_id = ? AND uhri IS NOT NULL
            ORDER BY timestamp ASC LIMIT ?
        """, (user_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except: return []

# ─────────────────────────────────────────────
#  AUTH HELPERS
# ─────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not creds: return None
    return SESSIONS.get(creds.credentials)

def require_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    uid = SESSIONS.get(creds.credentials) if creds else None
    if not uid: raise HTTPException(401, "Not authenticated")
    return uid

# ─────────────────────────────────────────────
#  LOAD ML MODELS
# ─────────────────────────────────────────────
def load_models():
    M = {}
    files = {
        'heart_model':    'models/heart_model.pkl',
        'heart_scaler':   'models/heart_scaler.pkl',
        'diabetes_model': 'models/diabetes_model.pkl',
        'diabetes_scaler':'models/diabetes_scaler.pkl',
        'disease_model':  'models/disease_model.pkl',
        'disease_le':     'models/disease_label_encoder.pkl',
        'symptom_list':   'models/symptom_list.pkl',
        'symptom_index':  'models/symptom_index.pkl',
        'desc_df':        'models/description_df.pkl',
        'prec_df':        'models/precautions_df.pkl',
        'meds_df':        'models/medications_df.pkl',
        'diets_df':       'models/diets_df.pkl',
        'workout_df':     'models/workout_df.pkl',
    }
    for key, path in files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                M[key] = pickle.load(f)
    return M

M = load_models()

def load_skin():
    R = {}
    for fname in ["skin_class_indices.pkl","class_indices.pkl"]:
        p = f"models/{fname}"
        if os.path.exists(p):
            with open(p,'rb') as f: R['class_indices'] = pickle.load(f)
            break
    if not TORCH_AVAILABLE: return R
    for mname, fname in [
        ("efficientnet","efficientnet.pth"),("efficientnet","efficientnet.path"),
        ("resnet","resnet.pth"),("resnet","resnet.path"),
        ("mobilenet","mobilenet.pth"),("mobilenet","mobilenet.path"),
    ]:
        p = f"models/{fname}"
        if not os.path.exists(p): continue
        try:
            if mname=="efficientnet":
                m=tvm.efficientnet_b0(weights=None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,8)
            elif mname=="resnet":
                m=tvm.resnet50(weights=None); m.fc=nn.Linear(m.fc.in_features,8)
            else:
                m=tvm.mobilenet_v2(weights=None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,8)
            st=torch.load(p,map_location='cpu')
            m.load_state_dict(st.get('model_state_dict',st.get('state_dict',st)))
            m.eval(); R['model']=m; R['model_name']=mname.capitalize(); break
        except: continue
    for fname in ["severity_model.pth","severity_model.path"]:
        p=f"models/{fname}"
        if not os.path.exists(p): continue
        try:
            sv=tvm.resnet18(weights=None); sv.fc=nn.Linear(sv.fc.in_features,3)
            st=torch.load(p,map_location='cpu')
            sv.load_state_dict(st.get('model_state_dict',st)); sv.eval(); R['severity_model']=sv
        except: pass
    return R

SKIN_M = load_skin()

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def rl(score):
    if score<.30: return "SAFE","#00e5c3"
    elif score<.60: return "MONITOR","#ffb347"
    elif score<.80: return "HIGH","#ff7043"
    return "CRITICAL","#ff4e6a"

def get_info(disease):
    def sg(key):
        df=M.get(key)
        if df is None: return []
        try:
            row=df[df.iloc[:,0].str.lower()==disease.lower()]
            if row.empty: return []
            return [str(v).strip() for v in row.iloc[0,1:].dropna().tolist() if str(v).strip() not in ('','nan')]
        except: return []
    return {'description':sg('desc_df'),'precautions':sg('prec_df'),
            'medications':sg('meds_df'),'diet':sg('diets_df'),'workout':sg('workout_df')}

# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────
class RegisterIn(BaseModel):
    username: str; email: str; password: str

class LoginIn(BaseModel):
    username: str; password: str

class HeartIn(BaseModel):
    age:float; sex:int; cp:int; trestbps:float; chol:float
    fbs:int; restecg:int; thalach:float; exang:int
    oldpeak:float; slope:int; ca:int; thal:int

class DiabetesIn(BaseModel):
    pregnancies:float; glucose:float; blood_pressure:float
    skin_thickness:float; insulin:float; bmi:float
    diabetes_pedigree:float; age:float

class DiseaseIn(BaseModel):
    symptoms: List[str]

class ReportIn(BaseModel):
    patient_name: str
    heart_prob: Optional[float] = None
    diabetes_prob: Optional[float] = None
    disease: Optional[str] = None
    uhri: Optional[float] = None

class ChatIn(BaseModel):
    message: str
    context: Optional[str] = None

# ─────────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────────
@app.post("/api/auth/register")
def register(data: RegisterIn):
    if len(data.username) < 3: raise HTTPException(400, "Username too short")
    if len(data.password) < 4: raise HTTPException(400, "Password too short")
    try:
        conn = get_db()
        conn.execute("INSERT INTO users (username,email,password_hash) VALUES (?,?,?)",
                     (data.username.strip(), data.email.strip(), hash_pw(data.password)))
        conn.commit()
        uid = conn.execute("SELECT id FROM users WHERE username=?", (data.username,)).fetchone()['id']
        conn.close()
        token = secrets.token_hex(32)
        SESSIONS[token] = uid
        return {"token": token, "username": data.username, "user_id": uid}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username or email already exists")

@app.post("/api/auth/login")
def login(data: LoginIn):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username=?", (data.username,)).fetchone()
    conn.close()
    if not user or not check_pw(data.password, user['password_hash']):
        raise HTTPException(401, "Invalid username or password")
    token = secrets.token_hex(32)
    SESSIONS[token] = user['id']
    return {"token": token, "username": user['username'], "user_id": user['id']}

@app.post("/api/auth/logout")
def logout(creds: HTTPAuthorizationCredentials = Depends(security)):
    if creds and creds.credentials in SESSIONS:
        del SESSIONS[creds.credentials]
    return {"status": "logged out"}

# ─────────────────────────────────────────────
#  GENERAL ROUTES
# ─────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "models": {
            "heart": "heart_model" in M,
            "diabetes": "diabetes_model" in M,
            "disease": "disease_model" in M,
            "skin": "model" in SKIN_M,
        },
        "shap": SHAP_AVAILABLE,
        "torch": TORCH_AVAILABLE,
        "db": os.path.exists(DB_PATH),
        "time": datetime.now().isoformat()
    }

@app.get("/api/symptoms")
def symptoms():
    if 'symptom_list' not in M: return {"symptoms": []}
    cleaned = []
    for s in M['symptom_list']:
        s = str(s).strip()
        if s in ('','nan') or s.replace('.','').replace('-','').isdigit(): continue
        cleaned.append(s.replace('_',' ').title())
    return {"symptoms": cleaned}

@app.get("/api/history")
def history(user_id: int = Depends(require_user)):
    return {"history": get_history(user_id)}

@app.get("/api/uhri-trend")
def uhri_trend(user_id: int = Depends(require_user)):
    return {"trend": get_uhri_trend(user_id)}

# ─────────────────────────────────────────────
#  HEART
# ─────────────────────────────────────────────
@app.post("/api/heart")
def heart(data: HeartIn, user_id: int = Depends(require_user)):
    if 'heart_model' not in M:
        raise HTTPException(503, "Heart model not found. Run notebook 01 first.")
    feat = np.array([[data.age,data.sex,data.cp,data.trestbps,data.chol,
                      data.fbs,data.restecg,data.thalach,data.exang,
                      data.oldpeak,data.slope,data.ca,data.thal]])
    sc = M['heart_scaler'].transform(feat)
    prob = float(M['heart_model'].predict_proba(sc)[0][1])
    label, color = rl(prob)

    shap_data = None
    fnames = ['Age','Sex','Chest Pain','Resting BP','Cholesterol','Fasting BS',
              'Rest ECG','Max HR','Exercise Angina','ST Depression','ST Slope','Vessels','Thal']
    if SHAP_AVAILABLE:
        try:
            ex = shap.TreeExplainer(M['heart_model'])
            sv = ex.shap_values(sc)
            vals = sv[1][0] if isinstance(sv,list) else sv[0]
            shap_data = sorted([{"feature":n,"value":float(v),"input":float(feat[0][i])}
                                 for i,(n,v) in enumerate(zip(fnames,vals))],
                               key=lambda x:abs(x['value']),reverse=True)[:10]
        except: pass

    save_prediction(user_id,'heart',heart_prob=prob,risk_lv=label)
    return {"probability":round(prob,4),"percentage":round(prob*100,1),
            "risk_level":label,"color":color,"shap":shap_data,
            "advice":"⚠️ Elevated cardiac risk. Consult a cardiologist." if prob>.5 else "✅ Low cardiac risk. Keep healthy habits!"}

# ─────────────────────────────────────────────
#  DIABETES
# ─────────────────────────────────────────────
@app.post("/api/diabetes")
def diabetes(data: DiabetesIn, user_id: int = Depends(require_user)):
    if 'diabetes_model' not in M:
        raise HTTPException(503, "Diabetes model not found. Run notebook 02 first.")
    feat = np.array([[data.pregnancies,data.glucose,data.blood_pressure,
                      data.skin_thickness,data.insulin,data.bmi,
                      data.diabetes_pedigree,data.age]])
    sc = M['diabetes_scaler'].transform(feat)
    prob = float(M['diabetes_model'].predict_proba(sc)[0][1])
    label, color = rl(prob)

    shap_data = None
    fnames = ['Pregnancies','Glucose','Blood Pressure','Skin Thickness',
              'Insulin','BMI','Diabetes Pedigree','Age']
    if SHAP_AVAILABLE:
        try:
            ex = shap.TreeExplainer(M['diabetes_model'])
            sv = ex.shap_values(sc)
            vals = sv[1][0] if isinstance(sv,list) else sv[0]
            shap_data = sorted([{"feature":n,"value":float(v),"input":float(feat[0][i])}
                                 for i,(n,v) in enumerate(zip(fnames,vals))],
                               key=lambda x:abs(x['value']),reverse=True)
        except: pass

    save_prediction(user_id,'diabetes',diabetes_prob=prob,risk_lv=label)
    return {"probability":round(prob,4),"percentage":round(prob*100,1),
            "risk_level":label,"color":color,"shap":shap_data,
            "advice":"⚠️ Elevated diabetes risk. Monitor glucose levels." if prob>.5 else "✅ Low diabetes risk. Maintain healthy diet!"}

# ─────────────────────────────────────────────
#  DISEASE
# ─────────────────────────────────────────────
@app.post("/api/disease")
def disease(data: DiseaseIn, user_id: int = Depends(require_user)):
    if 'disease_model' not in M:
        raise HTTPException(503, "Disease model not found. Run notebook 03 first.")
    sidx = M['symptom_index']; slist = M['symptom_list']
    vec = [0]*len(slist)
    for s in data.symptoms:
        s2 = s.strip().lower().replace(' ','_')
        if s2 in sidx: vec[sidx[s2]] = 1
    df = pd.DataFrame([vec], columns=slist)
    pred = M['disease_model'].predict(df)[0]
    proba = M['disease_model'].predict_proba(df)[0]
    name = M['disease_le'].inverse_transform([pred])[0]
    conf = float(proba[pred])
    top3 = [{"disease":M['disease_le'].inverse_transform([i])[0],"confidence":round(float(proba[i])*100,1)}
            for i in np.argsort(proba)[::-1][:3]]
    info = get_info(name)
    save_prediction(user_id,'disease',disease_name=name,confidence=conf*100)
    return {"disease":name,"confidence":round(conf*100,1),"top3":top3,"info":info}

# ─────────────────────────────────────────────
#  UHRI SAVE (called after both heart+diabetes done)
# ─────────────────────────────────────────────
class UhriIn(BaseModel):
    heart_prob: float; diabetes_prob: float

@app.post("/api/uhri")
def save_uhri(data: UhriIn, user_id: int = Depends(require_user)):
    uhri = data.heart_prob*0.6 + data.diabetes_prob*0.4
    label, _ = rl(uhri)
    save_prediction(user_id,'uhri',heart_prob=data.heart_prob,
                    diabetes_prob=data.diabetes_prob,uhri=uhri,risk_lv=label)
    return {"uhri":round(uhri,4),"percentage":round(uhri*100,1),"risk_level":label}

# ─────────────────────────────────────────────
#  SKIN
# ─────────────────────────────────────────────
CLASS_NAMES = ["Actinic keratosis","Atopic Dermatitis","Benign keratosis",
               "Dermatofibroma","Melanocytic nevus","Melanoma",
               "Squamous cell carcinoma","Tinea Ringworm Candidiasis"]

@app.post("/api/skin")
async def skin(file: UploadFile = File(...), user_id: int = Depends(require_user)):
    if 'model' not in SKIN_M:
        raise HTTPException(503, "Skin model not found. Place .pth files in models/")
    img = PILImage.open(io.BytesIO(await file.read())).convert("RGB")
    t = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(img).unsqueeze(0)
    names = CLASS_NAMES
    if 'class_indices' in SKIN_M:
        ci = SKIN_M['class_indices']
        if isinstance(ci,dict): names = [k for k,v in sorted(ci.items(),key=lambda x:x[1])]
    with torch.no_grad():
        pr = torch.softmax(SKIN_M['model'](t),dim=1)[0]
    tp,ti = torch.topk(pr,3)
    preds = [{"disease":names[i] if i<len(names) else f"Class {i}","confidence":round(float(p)*100,1)}
             for p,i in zip(tp.tolist(),ti.tolist())]
    sev = None
    if 'severity_model' in SKIN_M:
        with torch.no_grad():
            sev = ["Mild","Moderate","Severe"][torch.argmax(torch.softmax(SKIN_M['severity_model'](t),dim=1)[0]).item()]
    return {"predictions":preds,"severity":sev,"model":SKIN_M.get('model_name','CNN')}

# ─────────────────────────────────────────────
#  PDF REPORT
# ─────────────────────────────────────────────
@app.post("/api/pdf")
async def generate_pdf(data: ReportIn, creds: HTTPAuthorizationCredentials = Depends(security)):
    user_id = require_user(creds)
    if not REPORTLAB:
        raise HTTPException(500, "reportlab missing")
    from reportlab.pdfgen import canvas
    
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 800, "HealthFusion AI - Medical Report")
    c.drawString(100, 780, f"Patient: {data.patient_name}")
    
    y = 750
    if data.heart_prob is not None:
        c.drawString(100, y, f"Heart Risk: {data.heart_prob:.1f}%")
        y -= 20
    if data.diabetes_prob is not None:
        c.drawString(100, y, f"Diabetes Risk: {data.diabetes_prob:.1f}%")
        y -= 20
    if data.uhri is not None:
        c.drawString(100, y, f"UHRI Score: {data.uhri:.1f}")

    c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=report.pdf"
    })

# ─────────────────────────────────────────────
#  CHATBOT ENDPOINT
# ─────────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(data: ChatIn, creds: HTTPAuthorizationCredentials = Depends(security)):
    user_id = require_user(creds)
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key is not configured on the server.")
        
    system_prompt = f"""
You are the HealthFusion AI Hybrid Medical Assistant. 
You act as an intelligent medical report explainer.
Your job is to explain prediction results, discuss feature importance (like SHAP/LIME), and provide general lifestyle suggestions based on risk levels.
Context about the user's latest prediction: {data.context if data.context else 'No current prediction context.'}

STRICT INSTRUCTIONS:

1. NEVER prescribe medication.
2. NEVER diagnose unrelated topics outside of the provided context or general health wellness framing.
3. Keep your answers EXTREMELY SHORT, clear, and empathetic. Use bullet points and bold text for readability. Limit your response to 3-4 short sentences or bullets maximum! The user wants quick, scannable insights.
4. Say Asalam-o-Alaikum to the user.
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_prompt, data.message]
        )
        return {"reply": response.text}
    except Exception as e:
        print("Chat Error:", e)
        raise HTTPException(status_code=500, detail="Failed to generate response from AI.")


# ─────────────────────────────────────────────
#  SERVE FRONTEND
# ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root(): return FileResponse("static/index.html")

@app.get("/{path:path}")
def spa(path: str): return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
