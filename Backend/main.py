# Backend/main.py
import os
import sys
import uvicorn

# Ensure this directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Optional: load .env if you use it (no-op if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if __name__ == "__main__":
    # You can hardcode here, or override via env:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "true").lower() == "true"

    # IMPORTANT: point to the actual app: "api.server:app"
    uvicorn.run("api.server:app", host=host, port=port, reload=reload_flag)
