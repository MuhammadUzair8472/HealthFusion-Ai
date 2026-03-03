# HealthFusion AI v3 — Setup Guide

## ⚡ QUICKEST WAY (Windows)
Double-click `START.bat` — yeh sab kuch automatically karega!

## ⚡ QUICKEST WAY (Mac/Linux)
```bash
chmod +x start.sh
./start.sh
```

---

## Manual Setup (agar batch file kaam na kare)

### Step 1 — Install Python
https://python.org/downloads
**Important:** "Add Python to PATH" checkbox zaroor check karo!

### Step 2 — Install Packages
Open Command Prompt / Terminal in this folder, then:
```
pip install fastapi uvicorn[standard] python-multipart numpy pandas scikit-learn xgboost reportlab bcrypt pillow
```

### Step 3 — Copy Your Models
```
HealthFusionAI_v3/
└── models/
    ├── heart_model.pkl          ← copy from old project
    ├── heart_scaler.pkl
    ├── diabetes_model.pkl
    ├── diabetes_scaler.pkl
    ├── disease_model.pkl
    ├── disease_label_encoder.pkl
    ├── symptom_list.pkl
    ├── symptom_index.pkl
    ├── description_df.pkl
    ├── precautions_df.pkl
    ├── medications_df.pkl
    ├── diets_df.pkl
    ├── workout_df.pkl
    └── efficientnet.pth / resnet.pth  (optional - skin)
```

### Step 4 — Start Server
```
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5 — Open Browser
```
http://localhost:8000
```

---

## Troubleshooting

### "Connection Refused" in browser
→ Server is not running. Run Step 4 first.

### "ModuleNotFoundError: fastapi"
→ Run Step 2 again. Make sure pip works:
```
python -m pip install fastapi uvicorn[standard] python-multipart
```

### "Port already in use"
→ Change port: `python -m uvicorn api:app --port 8001`
→ Then open: http://localhost:8001

### Models not loading (API shows 0/4)
→ Copy .pkl files to the `models/` folder

---

## API Endpoints
| Endpoint | Method | Auth Required |
|----------|--------|--------------|
| `/api/health` | GET | No |
| `/api/auth/login` | POST | No |
| `/api/auth/register` | POST | No |
| `/api/heart` | POST | Yes (token) |
| `/api/diabetes` | POST | Yes (token) |
| `/api/disease` | POST | Yes (token) |
| `/api/skin` | POST | Yes (token) |
| `/api/report` | POST | Yes (token) |
| `/api/history` | GET | Yes (token) |
| `/api/uhri-trend` | GET | Yes (token) |
