from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from peewee import DoesNotExist
import uuid
import datetime

from .models import ConversionTask
from .pipeline import process
from .utils.pdf_exporter import (
    export_to_pdf,
)
from .utils.word_exporter import export_to_word

router = APIRouter()


@router.post("/convert")
async def convert_file(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    task_id = str(uuid.uuid4())
    ConversionTask.create(
        id=task_id,
        status="processing",
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
    )
    background_tasks.add_task(process, task_id, await file.read())
    return {"id": task_id, "status": "processing"}


@router.post("/convert/{scheme_id}/{meeting_id}")
async def convert_file_with_id(
    background_tasks: BackgroundTasks,
    scheme_id: str,
    meeting_id: str,
    file: UploadFile = File(...),
):
    """Convert file with optional scheme_id and meeting_id."""

    if not scheme_id or not meeting_id:
        return JSONResponse(
            status_code=400,
            content={"error": "scheme_id and meeting_id are required"},
        )

    task_id = str(uuid.uuid4())
    ConversionTask.create(
        id=task_id,
        status="processing",
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
    )
    background_tasks.add_task(
        process, task_id, await file.read(), scheme_id, meeting_id
    )
    return {"id": task_id, "status": "processing"}


@router.get("/result/{task_id}")
def get_result(task_id: str):
    try:
        task = ConversionTask.get(ConversionTask.id == task_id)
    except DoesNotExist:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return {
        "id": task.id,
        "status": task.status,
        "result_json": task.result_json if task.status == "done" else None,
        "error_message": (
            task.error_message if task.status == "failed" else None
        ),
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


@router.get("/tasks")
def list_tasks():
    tasks = ConversionTask.select()
    return [
        {
            "id": t.id,
            "status": t.status,
            "created_at": t.created_at,
            "updated_at": t.updated_at,
        }
        for t in tasks
    ]


@router.get("/export/pdf/{task_id}")
def export_pdf(task_id: str):
    try:
        task = ConversionTask.get(ConversionTask.id == task_id)
    except DoesNotExist:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    if not task.result_json:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    pdf_bytes = export_to_pdf(task.result_json)
    return Response(
        pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={task_id}.pdf"},
    )


@router.get("/export/word/{task_id}")
def export_word(task_id: str):
    try:
        task = ConversionTask.get(ConversionTask.id == task_id)
    except DoesNotExist:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    if not task.result_json:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    word_bytes = export_to_word(task.result_json)
    return Response(
        word_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={task_id}.docx"},
    )
