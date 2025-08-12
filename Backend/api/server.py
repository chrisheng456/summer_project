# Backend/api/server.py
from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# relative import: the pipeline package is under api/app/...
from .app.pipeline import process, process_pipeline

app = FastAPI(title="Meeting Pipeline API")

# Allow front-end to call (relax in dev; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# A) Run pipeline synchronously and return result (no DB task)
@app.post("/pipeline/run")
async def run_pipeline(file: UploadFile = File(...)):
    # Read bytes into memory; your process_pipeline accepts raw bytes
    content = await file.read()
    info = process_pipeline(content)
    # Return the final JSON that front-end needs to consume
    return {"ok": True, "result": info.customer_meeting_detail}

# B) Run as a background task and let `process(...)` update DB
@app.post("/pipeline/run-task")
async def run_pipeline_task(
    file: UploadFile = File(...),
    task_id: int = 0,
    scheme_id: str | None = None,
    meeting_id: str | None = None,
    bg: BackgroundTasks = None,
):
    content = await file.read()
    # `process` will handle DB status, error and result_json update
    bg.add_task(process, task_id, content, scheme_id, meeting_id)
    return {"ok": True, "task_id": task_id}
