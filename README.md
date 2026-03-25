# 🏥 HealthFusion AI v2

> **An AI-powered multi-disease health prediction platform combining Classical ML, Deep Learning (PyTorch), and Generative AI (Gemini) into a single unified web application.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Models](#ml-models)
- [API Endpoints](#api-endpoints)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [Authentication System](#authentication-system)
- [Database](#database)
- [Frontend](#frontend)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)

---

## 🌟 Overview

**HealthFusion AI** is a comprehensive medical AI platform that predicts and analyzes multiple diseases using a combination of:

- **Classical Machine Learning** – Scikit-learn models for Heart Disease and Diabetes prediction
- **Deep Learning (PyTorch/CNN)** – EfficientNet-based models for Skin Disease and Brain Tumor classification
- **Symptom-based NLP** – Multi-label disease prediction using encoded symptom vectors
- **Generative AI (Gemini 2.5 Flash)** – AI chatbot that explains predictions and gives health advice
- **Explainable AI (SHAP)** – Feature importance for transparent ML predictions
- **PDF Reports** – Downloadable patient health reports via ReportLab

The platform features full **user authentication**, **prediction history tracking**, **UHRI (Unified Health Risk Index)** scoring, and a modern Single Page Application (SPA) frontend.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🫀 **Heart Disease Prediction** | Logistic Regression/XGBoost with 13 clinical features and SHAP explainability |
| 🩸 **Diabetes Prediction** | ML classifier with 8 biometric features and SHAP feature importance |
| 🤒 **Symptom Disease Checker** | Multi-symptom encoder predicting from 41+ diseases with Top-3 results |
| 🧠 **Brain Tumor Classification** | EfficientNet-B0 CNN classifying Glioma, Meningioma, Pituitary, or No Tumor |
| 🦠 **Skin Disease Detection** | EfficientNet CNN classifying 8 skin conditions with severity rating |
| 🤖 **AI Medical Chatbot** | Gemini 2.5 Flash powered chatbot for explaining predictions |
| 📊 **UHRI Score** | Unified Health Risk Index = Heart(60%) + Diabetes(40%) composite score |
| 📄 **PDF Reports** | Downloadable medical report with all prediction results |
| 🔐 **Auth System** | JWT-style token auth with SQLite user management |
| 📈 **History Tracking** | Full prediction history and UHRI trend charts per user |

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** | REST API framework (v0.111.0) |
| **Uvicorn** | ASGI server for FastAPI |
| **SQLite** | Lightweight database for users & predictions |
| **Python-Multipart** | File upload support (MRI/skin images) |
| **Bcrypt** | Secure password hashing |

### Machine Learning
| Library | Purpose |
|---|---|
| **Scikit-learn** | Heart & Diabetes ML models |
| **XGBoost** | Gradient boosted tree models |
| **PyTorch + Torchvision** | Brain Tumor & Skin CNN models |
| **SHAP** | Explainable AI feature importance |
| **NumPy + Pandas** | Data processing and feature engineering |
| **Pillow (PIL)** | Image preprocessing for CNN models |

### AI & Reporting
| Library | Purpose |
|---|---|
| **Google Generative AI (`google-genai`)** | Gemini 2.5 Flash chatbot integration |
| **ReportLab** | PDF medical report generation |

### Frontend
| Technology | Purpose |
|---|---|
| **HTML5 / CSS3 / JavaScript** | Single Page Application (no framework) |
| **Vanilla JS Fetch API** | API calls to FastAPI backend |

---

## 📁 Project Structure

```
HealthFusionAI_v2/
│
├── api.py                        # 🔑 Main FastAPI backend (all routes & ML logic)
├── requirements.txt              # Python dependencies
├── START.bat                     # Windows one-click setup & launch script
├── Procfile                      # Deployment config (e.g., Heroku/Render)
├── healthfusion.db               # SQLite database (auto-created)
├── .env                          # Environment variables (Gemini API key)
│
├── static/
│   ├── index.html                # 🌐 Frontend SPA (all UI in one file)
│   └── chatbot.json              # Chatbot conversation data/config
│
├── models/                       # 🤖 All trained ML model files
│   ├── EfficientNet_v2_Phase1_best.pth  # Brain Tumor CNN (EfficientNet-B0, 4 classes)
│   ├── efficientnet.pth          # Skin Disease CNN (EfficientNet-B0, 8 classes)
│   ├── severity_model.pth        # Skin Severity CNN (ResNet18, 3 classes)
│   ├── heart_model.pkl           # Heart Disease classifier
│   ├── heart_scaler.pkl          # Heart feature scaler
│   ├── heart_features.pkl        # Heart feature names
│   ├── diabetes_model.pkl        # Diabetes classifier
│   ├── diabetes_scaler.pkl       # Diabetes feature scaler
│   ├── diabetes_features.pkl     # Diabetes feature names
│   ├── disease_model.pkl         # Multi-disease symptom classifier
│   ├── disease_label_encoder.pkl # Disease name encoder
│   ├── symptom_list.pkl          # Full symptom vocabulary
│   ├── symptom_index.pkl         # Symptom-to-index mapping
│   ├── description_df.pkl        # Disease descriptions
│   ├── precautions_df.pkl        # Disease precautions
│   ├── medications_df.pkl        # Medications info
│   ├── diets_df.pkl              # Diet recommendations
│   └── workout_df.pkl            # Workout recommendations
│
├── dataset/                      # Training datasets (notebooks)
├── notebook/                     # Jupyter notebooks for model training
├── Output Images/                # Sample model output images
├── hfv2/                         # Python virtual environment
│
├── test_brain.py                 # Test script for Brain Tumor API
├── test_keys.py                  # Test script for API keys
└── test_load.py                  # Test script for model loading
```

---

## 🤖 ML Models

### 1. 🫀 Heart Disease Model
- **Type:** Classification (Logistic Regression / Random Forest / XGBoost)
- **Input:** 13 clinical features
- **Features:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression, ST Slope, Number of Vessels, Thalassemia
- **Output:** Probability score (0–1) + Risk Level (SAFE / MONITOR / HIGH / CRITICAL)
- **Explainability:** SHAP TreeExplainer showing top contributing features

### 2. 🩸 Diabetes Model
- **Type:** Classification (Random Forest / XGBoost)
- **Input:** 8 biometric features
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Output:** Probability score (0–1) + Risk Level
- **Explainability:** SHAP feature importance values

### 3. 🤒 Disease Symptom Checker
- **Type:** Multi-class Classification
- **Input:** List of symptom strings (e.g., `["fever", "headache", "fatigue"]`)
- **Encoding:** Binary vector encoding over full symptom vocabulary
- **Output:** Disease name, confidence %, Top-3 predictions, plus Description / Precautions / Medications / Diets / Workout

### 4. 🧠 Brain Tumor Model
- **Architecture:** EfficientNet-B0 (custom classifier head)
- **Classes:** 4 — `Glioma`, `Meningioma`, `No Tumor`, `Pituitary`
- **Input:** MRI scan image (JPEG/PNG), resized to 224×224
- **Classifier Head:**
  ```
  Dropout(0.2) → Linear(1280→256) → BatchNorm → ReLU → Dropout(0.5) → Linear(256→4)
  ```
- **Output:** Top-4 predictions with confidence scores + original image preview
- **Model File:** `models/EfficientNet_v2_Phase1_best.pth`

### 5. 🦠 Skin Disease Model
- **Architecture:** EfficientNet-B0 (8-class output)
- **Classes:** Actinic Keratosis, Atopic Dermatitis, Benign Keratosis, Dermatofibroma, Melanocytic Nevus, Melanoma, Squamous Cell Carcinoma, Tinea/Ringworm/Candidiasis
- **Severity Model:** ResNet18 (3-class) — Mild / Moderate / Severe
- **Input:** Skin image (JPEG/PNG), resized to 224×224
- **Output:** Top-3 disease predictions + severity rating

### 🎯 Risk Level System (UHRI)
The **Unified Health Risk Index (UHRI)** combines heart and diabetes risk:

```
UHRI = Heart_Probability × 0.6 + Diabetes_Probability × 0.4
```

| Score Range | Risk Level | Color |
|---|---|---|
| 0.00 – 0.30 | ✅ SAFE | Cyan `#00e5c3` |
| 0.30 – 0.60 | ⚠️ MONITOR | Orange `#ffb347` |
| 0.60 – 0.80 | 🔴 HIGH | Red-Orange `#ff7043` |
| 0.80 – 1.00 | 🚨 CRITICAL | Red `#ff4e6a` |

---

## 🔌 API Endpoints

### Authentication
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/auth/register` | Register new user | No |
| `POST` | `/api/auth/login` | Login, returns bearer token | No |
| `POST` | `/api/auth/logout` | Logout, invalidate token | Yes |

### Health & Utilities
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/health` | System health check, model status | No |
| `GET` | `/api/symptoms` | Get full list of available symptoms | No |

### Predictions
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/heart` | Heart disease prediction | Yes |
| `POST` | `/api/diabetes` | Diabetes prediction | Yes |
| `POST` | `/api/disease` | Symptom-based disease prediction | Yes |
| `POST` | `/api/skin` | Skin disease detection (image upload) | Yes |
| `POST` | `/api/brain` | Brain tumor classification (image upload) | Yes |
| `POST` | `/api/uhri` | Calculate & save UHRI composite score | Yes |

### History & Reports
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/history` | Get user prediction history | Yes |
| `GET` | `/api/uhri-trend` | Get UHRI trend data for chart | Yes |
| `POST` | `/api/pdf` | Generate downloadable PDF medical report | Yes |

### AI Chatbot
| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/chat` | Chat with Gemini AI medical assistant | Yes |

### Frontend
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve frontend SPA (index.html) |
| `GET` | `/{path}` | SPA fallback routing |

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.9+** (Tested on Python 3.10/3.11)
- **pip** (comes with Python)
- **Git** (optional, for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/HealthFusionAI_v2.git
cd HealthFusionAI_v2
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** PyTorch CPU-only install is faster if you don't have a GPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

> Get your Gemini API key from: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### Step 5: Ensure Models Are in Place
All model files (`.pth` and `.pkl` files) must be in the `models/` directory. The key model files are:
- `models/EfficientNet_v2_Phase1_best.pth` — Brain Tumor model
- `models/efficientnet.pth` — Skin Disease model
- `models/severity_model.pth` — Skin Severity model
- `models/heart_model.pkl` — Heart Disease model
- `models/diabetes_model.pkl` — Diabetes model
- `models/disease_model.pkl` — Symptom Disease model

---

## 🚀 Running the Project

### Option A: One-Click Windows Launch (Recommended)
Double-click `START.bat` — it will:
1. Check Python installation
2. Install all required packages
3. Start the server at `http://localhost:8000`

### Option B: Manual Start
```bash
# Activate virtual environment first
venv\Scripts\activate         # Windows
source venv/bin/activate      # macOS/Linux

# Start the FastAPI server
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Option C: Direct Python
```bash
python api.py
```

### Access the App
Once started, open your browser and go to:
```
http://localhost:8000
```

The API documentation (Swagger UI) is available at:
```
http://localhost:8000/docs
```

---

## 🔐 Authentication System

HealthFusion AI uses a **token-based authentication** system:

1. **Registration** → User registers with username, email, password
2. **Login** → Returns a `Bearer Token` (64-char hex secret)
3. **Authenticated Requests** → All prediction endpoints require:
   ```
   Authorization: Bearer <token>
   ```
4. **Sessions** → Stored in-memory (`SESSIONS` dict: `token → user_id`)
5. **Passwords** → Hashed with `bcrypt` (falls back to SHA-256 if bcrypt unavailable)
6. **Logout** → Token removed from SESSIONS dict

> ⚠️ Sessions are **in-memory** — they reset on server restart. For production, use Redis or a persistent session store.

---

## 🗄️ Database

HealthFusion AI uses **SQLite** (`healthfusion.db`) with WAL mode for concurrent access.

### Tables

**`users`**
```sql
id          INTEGER PRIMARY KEY AUTOINCREMENT
username    TEXT UNIQUE NOT NULL
email       TEXT UNIQUE NOT NULL
password_hash TEXT NOT NULL
created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

**`predictions`**
```sql
id            INTEGER PRIMARY KEY AUTOINCREMENT
user_id       INTEGER (FK → users.id)
pred_type     TEXT          -- 'heart', 'diabetes', 'disease', 'uhri'
heart_prob    REAL
diabetes_prob REAL
uhri          REAL
risk_level    TEXT
disease_name  TEXT
confidence    REAL
timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

The database is **auto-initialized** on first run via `init_db()`.

---

## 🌐 Frontend

The frontend is a **Single Page Application (SPA)** served from `static/index.html`.

### Key Sections
| Section | Description |
|---|---|
| **Dashboard** | Overview, UHRI gauge, quick stats, history chart |
| **Heart Check** | 13-field form → risk prediction + SHAP chart |
| **Diabetes Check** | 8-field form → risk prediction + SHAP chart |
| **Disease Finder** | Symptom multi-select → disease prediction + info cards |
| **Brain Scan** | MRI image upload → tumor classification results |
| **Skin Analysis** | Skin image upload → disease + severity |
| **AI Chatbot** | Gemini-powered health assistant |
| **PDF Report** | Download comprehensive health report |
| **Login / Register** | Auth modals, session management |

### Frontend Features
- 🎨 Dark mode medical UI with glassmorphism effects
- 📱 Fully responsive design
- 📊 Interactive risk gauges and SHAP bar charts
- 🔄 Real-time UHRI trend visualization
- 💬 Chat interface with typing animations
- 🔒 Token-based auth state management in localStorage

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | Google Gemini API key for the AI chatbot |

Create `.env` file in project root:
```
GEMINI_API_KEY=AIza...your_key_here
```

---

## 🚢 Deployment

### Procfile (Render / Heroku)
The `Procfile` is already configured:
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

### Deploy to Render.com
1. Push code to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variable: `GEMINI_API_KEY=your_key`
6. Upload model files to a persistent disk or use cloud storage

### ⚠️ Production Notes
- Replace in-memory `SESSIONS` with **Redis** for persistent sessions
- Store ML models on a **persistent disk** (not ephemeral storage)
- Add **rate limiting** to prevent API abuse
- Use **HTTPS** in production (TLS via Nginx reverse proxy)
- The SQLite database should use a **persistent volume** in containerized deployments

---

## 🧪 Testing

Run the included test scripts:
```bash
# Test Brain Tumor API
python test_brain.py

# Test model loading
python test_load.py

# Test API keys
python test_keys.py
```

---

## 📦 Key Dependencies Summary

```
fastapi==0.111.0         # Web framework
uvicorn[standard]==0.30.1 # ASGI server
python-multipart==0.0.9  # File uploads
torch + torchvision      # Deep learning (CNN models)
scikit-learn             # Classical ML models
xgboost                  # Gradient boosting
numpy + pandas           # Data processing
pillow                   # Image processing
shap                     # Explainable AI
bcrypt                   # Password hashing
reportlab                # PDF generation
google-genai             # Gemini AI chatbot
python-dotenv            # .env file support
```

---

## 👨‍💻 Author

**Muhammad Uzair**  
HealthFusion AI v2 — An AI-powered multi-disease health prediction platform

---

## 📄 License

This project is for educational and research purposes. All ML models are trained on publicly available medical datasets.

> ⚠️ **Medical Disclaimer:** This application is for informational purposes only. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.
