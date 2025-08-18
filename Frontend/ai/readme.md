# ai_content_edit 

This module exposes a content editing / rewriting API backed by Azure OpenAI, implemented with FastAPI.
It is designed to be easily embedded in different projects via environment-based configuration and a small, modular codebase.

---

## Project Structure

```text
PythonProject/
│
├── env/
│   └── .env                    # Environment variable configuration file
│
├── modules/
│   └── ai_content_edit/
│       ├── __init__.py         # Package declaration file
│       ├── chat_api.py         # FastAPI service interface (API entry point)
│       └── __init__.py         # Package initialization file (can be left as-is if not needed)
│
├── tools/
│   ├── azure_client.py         # Azure OpenAI client wrapper (connects to OpenAI)
│   └── get_env.py              # Helper functions for reading environment variables
│
├── readme.md                   # Documentation
├── requirements.txt            # Python dependency list
└── venv/                       # Virtual environment (auto-generated)
```
## File Descriptions
```
• env/.env
  Environment variable configuration file that stores sensitive values (e.g., API key, endpoint).
  Do NOT commit this file to your Git repository.

• modules/ai_content_edit/azure_client.py
  Azure OpenAI client wrapper that reads credentials and settings from environment variables. (Utility)

• modules/ai_content_edit/chat_api.py
  FastAPI main service that exposes the POST /api/chat endpoint and bridges to the Azure OpenAI Chat API. (Module functionality)

• modules/ai_content_edit/tools/get_env.py (Utility)
  General-purpose helpers for reading environment variables (useful if you want to reuse the logic).

• modules/ai_content_edit/__init__.py (Package declaration)
  Package declaration file; can be used for initialization imports, or left empty.

• requirements.txt (Dependencies)
  List of required Python packages, such as fastapi, uvicorn, openai, python-dotenv, etc.
```

## Usage

### 1.	Install dependencies
 `pip install -r requirements.txt`
 
### 2.	Configure .env
Create env/.env (do not commit this file; add .env to your .gitignore so it won’t be pushed):

``` 
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://xxxxxx.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o

You can add other model deployments later as needed.
```
### 3.	Start the service
`root：uvicorn modules.ai_content_edit.chat_api:app --reload`

### 4.	API 
	•	POST /api/chat
	•	Request body:： "messages": [
    {"role": "system", "content": "You are a content editing assistant."},
    {"role": "user", "content": "Please summarize the following content..."}] 

	•	Response body:{"reply": "This is the AI's reply."}
  

