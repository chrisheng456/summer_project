# Backend/api/app/api.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import uuid
import datetime

# 你项目里已有的导入，按需保留
from .models import ConversionTask
from .pipeline import process_pipeline as process
from .utils.pdf_exporter import export_to_pdf
from .utils.word_exporter import export_to_word
from .utils.pp_client import PPClient  # 用于调用客户 API

router = APIRouter()
security = HTTPBearer()          # ✅ Swagger 顶部会出现 Authorize 按钮
pp = PPClient()                  # 客户 API client 实例


# ---------------------------
# 登录：不需要 Bearer
# ---------------------------
@router.post("/auth/login")
async def login(body: dict):
    """
    body: { "username": "...", "password": "..." }
    返回示例: { "ok": true, "token": "...", "meetings": [...] }
    """
    username = body.get("username", "")
    password = body.get("password", "")

    # 你现有的登录逻辑（按你 PPClient 的方法名适配）
    token = pp.login(username=username, password=password)

    # 登录后顺便拉一次会议列表（可选）
    meetings = pp.list_meetings(token=token)

    return {"ok": True, "token": token, "meetings": meetings}


# ---------------------------
# 需要鉴权的接口（示例：会议列表/详情）
# ---------------------------
@router.get("/customer/meetings")
def list_meetings(creds: HTTPAuthorizationCredentials = Depends(security)):
    """
    使用 Authorize 里填的 token（纯 token）调用客户 API
    """
    token = creds.credentials  # 这里拿到的是纯 token，不带 "Bearer "
    meetings = pp.list_meetings(token=token)  # 按你的方法名来
    return meetings


@router.get("/customer/meetings/{meeting_id}")
def get_meeting_detail(meeting_id: str,
                       creds: HTTPAuthorizationCredentials = Depends(security)):
    token = creds.credentials
    detail = pp.get_meeting_detail(meeting_id=meeting_id, token=token)  # 按你的方法名来
    return detail


# ---------------------------
# 文件转换 / 导出（原有接口，示例保留）
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
