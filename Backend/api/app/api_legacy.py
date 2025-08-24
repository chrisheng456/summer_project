from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import uuid
import datetime

from .models import ConversionTask
from .pipeline import process_pipeline as process
from .utils.pdf_exporter import export_to_pdf
from .utils.word_exporter import export_to_word
from .utils.pp_client import PPClient

router = APIRouter()
security = HTTPBearer()
pp = PPClient()


@router.post("/auth/login")
async def login(body: dict):
    """
    Login endpoint.

    Expected body:
        {
            "username": "...",
            "password": "..."
        }

    Response example:
        {
            "ok": true,
            "token": "...",
            "meetings": [...]
        }
    """
    username = body.get("username", "")
    password = body.get("password", "")

    token = pp.login(username=username, password=password)

    # Fetching meeting list immediately after login
    meetings = pp.list_meetings(token=token)

    return {"ok": True, "token": token, "meetings": meetings}


# ---------------------------
# Auth-protected endpoints (meeting list / details)
# ---------------------------
@router.get("/customer/meetings")
def list_meetings(creds: HTTPAuthorizationCredentials = Depends(security)):
    """
    Fetch customer meeting list using the provided token.
    The token comes from the `Authorize` header (without the "Bearer " prefix).
    """
    token = creds.credentials
    meetings = pp.list_meetings(token=token)
    return meetings


@router.get("/customer/meetings/{meeting_id}")
def get_meeting_detail(meeting_id: str,
                       creds: HTTPAuthorizationCredentials = Depends(security)):
    token = creds.credentials
    detail = pp.get_meeting_detail(meeting_id=meeting_id, token=token)
    return detail


# ---------------------------
# File conversion / export endpoints
# ---------------------------
@router.post("/convert")
async def convert_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    ConversionTask.create(
        id=task_id,
        status="processing",
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
    )
    background_tasks.add_task(process, task_id, await file.read())
    return {"id": task_id, "status": "processing"}


@router.get("/export/pdf/{task_id}")
def export_pdf(task_id: str):
    pdf_bytes = export_to_pdf(task_id)
    return Response(content=pdf_bytes, media_type="application/pdf")


@router.get("/export/docx/{task_id}")
def export_docx(task_id: str):
    docx_bytes = export_to_word(task_id)
    headers = {"Content-Disposition": f'attachment; filename="{task_id}.docx"'}
    return Response(content=docx_bytes,
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers=headers)
