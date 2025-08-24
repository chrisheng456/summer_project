
# Project Startup Guide

This project consists of three parts:

- **Frontend/minutes**: Frontend (Vue 3 + Vite)
- **Frontend/ai**: AI microservice (Python + FastAPI/Uvicorn)
- **Backend**: Backend (Python + FastAPI/Uvicorn)

**Default Ports**

- AI Service: `http://127.0.0.1:54232`
- Backend: `http://127.0.0.1:8000`

---

## Directory Structure (Brief)

```
project-root/
├─ Frontend/
│ ├─ minutes/ # Vue frontend
│ └─ ai/ # Python AI microservice (Uvicorn)
├─ Backend/ # Python backend (FastAPI)
└─ README.md
```


---

## I. Prerequisites

- **Node.js ≥ 18** (for `Frontend/minutes`)
- **Python 3.11** (for `Frontend/ai` and `Backend`)
- Windows or macOS/Linux

---

## II. Frontend/minutes (Frontend)

### 1) Install Dependencies
```bash
cd Frontend/minutes
npm install
```
### 2) Environment Variables

You need to configure the backend/AI endpoints.

```
# Used by the frontend (example)
VITE_API_BASE=http://127.0.0.1:8000
VITE_AI_BASE=http://127.0.0.1:54232
```
Note: In the current Vite configuration, the AI port is fixed to http://127.0.0.1:54232. Make sure the AI service is started on this port as described below.

### 3) Start (Development Mode)
```
npm run dev

```
Open the local URL shown in the terminal (typically http://127.0.0.1:5173 or a similar port).

---

## III. Frontend/ai (AI Microservice)
### 1) Create and Activate a Virtual Environment

```
cd Frontend/ai
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2) Install Dependencies

```
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3) Create .env
Create a .env file under Frontend/ai

```
# Azure OpenAI
AZURE_OPENAI_API_KEY="***********"
AZURE_OPENAI_ENDPOINT="https://<your-cognitive>.cognitiveservices.azure.com/"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"  # API version
AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```

### 4) Start the AI Service (Port 54232)
The minutes frontend expects the AI service at http://127.0.0.1:54232. 
Use the following command (do not change the port unless you also update the frontend configuration):

```
uvicorn modules.ai_content_edit.chat_api:app --reload --host 127.0.0.1 --port 54232
```

After a successful start, the service is available at http://127.0.0.1:54232.

---

## IV. Backend
### 1) Create and Activate a Virtual Environment
```
cd Backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2) Install Dependencies

```
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

```
### 3) Create .env

```
# Customer system API authentication
CUSTOMER_API_USERNAME=ruixiong
CUSTOMER_API_PASSWORD=***************   # enter the real password
CUSTOMER_API_BASE_URL=https://pensionpal2test.azurewebsites.net/api/Logon

# Azure Speech Service
AZURE_SPEECH_KEY=*********************
AZURE_SPEECH_REGION=uksouth

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=...
AZURE_STORAGE_CONTAINER=cunchu

# Hugging Face (if used)
HUGGINGFACE_TOKEN=hf_*******************

# Service runtime configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
```

### 4) Start the Backend (Default Port 8000)

Option A: If the project has main.py
```
python main.py
```

Option B: Start directly with Uvicorn (recommended)
```
uvicorn api.server:app --reload --host 127.0.0.1 --port 8000
```
---
## VI. Quick Troubleshooting

ModuleNotFoundError: Make sure you ran pip install -r requirements.txt within the correct directory’s virtual environment.

Pydantic ValidationError (missing fields): Check that .env exists, variable names are correct, and values are provided.

Port not reachable: The AI service must run on 127.0.0.1:54232 and the backend on 127.0.0.1:8000 (unless you consistently update the frontend Vite configuration).
