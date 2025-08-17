# Backend/api/server.py
from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Optional

from fastapi import (
    FastAPI, UploadFile, File, Depends, HTTPException, Body, Response, Query
)
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .app.pipeline import process_pipeline
from .app.utils.pp_client import PPClient
from .app.utils.pdf_exporter import export_pdf_from_payload
from .app.utils.word_exporter import export_docx_from_payload

# 内存缓存：按 token / (token, scheme, meeting) 存最近一次结果
_LAST_DETAIL_BY_KEY: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_LAST_DETAIL_BY_TOKEN: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Meeting Pipeline API")
security = HTTPBearer(auto_error=True)


def _safe_len(x: Any) -> int:
    try:
        return len(x) if x is not None else 0
    except Exception:
        return 0


@app.get("/health")
def health():
    return {"ok": True}


# -------------- 登录 --------------
@app.post("/auth/login")
def login(
    payload: Dict[str, str] = Body(..., example={"username": "ruixiong", "password": "******"}),
    include_meetings: bool = Query(True, description="是否在登录后立即返回会议列表"),
):
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="username/password required")

    try:
        client = PPClient()
        token = client.login(username=username, password=password)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"remote login failed: {e}")

    resp: Dict[str, Any] = {"ok": True, "token": token}
    if include_meetings:
        try:
            client = PPClient(bearer_token=token)
            resp["meetings"] = client.list_meetings()
        except Exception as e:
            resp["meetings_error"] = str(e)
    return resp


# -------------- 会议列表 --------------
@app.get("/customer/meetings")
def list_meetings(creds: HTTPAuthorizationCredentials = Depends(security)):
    token = creds.credentials
    try:
        client = PPClient(bearer_token=token)
        meetings = client.list_meetings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch meetings failed: {e}")
    return {"ok": True, "meetings": meetings}


# -------------- 会议详情 --------------
@app.get("/customer/meetings/{scheme_id}/{meeting_id}")
def get_meeting_detail(
    scheme_id: str,
    meeting_id: str,
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    token = creds.credentials
    try:
        client = PPClient(bearer_token=token)
        detail = client.meeting_detail(scheme_id, meeting_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch meeting detail failed: {e}")
    return {"ok": True, "detail": detail}


# -------------- 一体化分析 --------------
@app.post("/pipeline/analyze")
async def analyze(
    scheme_id: str = Query(..., description="Scheme ID"),
    meeting_id: str = Query(..., description="Meeting ID"),
    file: UploadFile = File(..., description="会议原始音频文件"),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    bearer_token = creds.credentials
    data = await file.read()

    try:
        info = process_pipeline(
            input_file_content=data,
            scheme_id=scheme_id,
            meeting_id=meeting_id,
            bearer_token=bearer_token,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")

    stt_lines = getattr(info, "transcription", None)
    cleaned_lines = getattr(info, "cleaned_transcription", None)
    diarization = getattr(info, "speaker_segments", None) or getattr(info, "diarization", None)
    classification = getattr(info, "classified_lines", None)
    label_counts = getattr(info, "label_counts", None)
    summary = getattr(info, "summary", None)
    meeting_detail = getattr(info, "customer_meeting_detail", None)

    # 缓存最近一次结果（用于 /export/* 无需再传 body）
    if meeting_detail:
        token = bearer_token
        _LAST_DETAIL_BY_TOKEN[token] = meeting_detail
        _LAST_DETAIL_BY_KEY[(token, str(scheme_id), str(meeting_id))] = meeting_detail

    payload = {
        "ok": True,
        "customer_meeting_detail": meeting_detail,
        "speech_to_text": {"total": _safe_len(stt_lines), "lines": stt_lines},
        "data_cleaning": {"total": _safe_len(cleaned_lines), "lines": cleaned_lines},
        "speaker_diarization": {"total": _safe_len(diarization), "segments": diarization},
        "text_classification": {
            "total": _safe_len(classification),
            "lines": classification,
            "label_counts": label_counts,
        },
        "text_summary": summary,
    }
    return jsonable_encoder(payload)


# ---------------- 导出：工具函数 ----------------
def _try_extract_detail(payload: Any) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    try:
        if isinstance(payload, (bytes, bytearray, str)):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            return None
        detail = payload.get("customer_meeting_detail") if "customer_meeting_detail" in payload else payload
        return detail if isinstance(detail, dict) and isinstance(detail.get("agenda"), list) else None
    except Exception:
        return None


def _get_cached_detail_or_400(
    creds: HTTPAuthorizationCredentials,
    scheme_id: Optional[str],
    meeting_id: Optional[str],
) -> Dict[str, Any]:
    token = creds.credentials
    if scheme_id and meeting_id:
        key = (token, str(scheme_id), str(meeting_id))
        if key in _LAST_DETAIL_BY_KEY:
            return _LAST_DETAIL_BY_KEY[key]
    if token in _LAST_DETAIL_BY_TOKEN:
        return _LAST_DETAIL_BY_TOKEN[token]
    raise HTTPException(status_code=400, detail="no cached result found; please run /pipeline/analyze first")


# -------------- 导出 PDF --------------
@app.post("/export/pdf")
def export_pdf(
    result: Optional[Dict[str, Any]] = Body(
        None,
        description="可选：完整响应或 customer_meeting_detail；为空或不合法将使用最近一次分析结果",
    ),
    scheme_id: Optional[str] = Query(None, description="同账号多会议时指定 scheme_id"),
    meeting_id: Optional[str] = Query(None, description="同账号多会议时指定 meeting_id"),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        detail = _try_extract_detail(result) or _get_cached_detail_or_400(creds, scheme_id, meeting_id)
        pdf_bytes = export_pdf_from_payload(detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"export pdf failed: {e}")

    filename = (detail.get("name") or "meeting_minutes").replace('"', "").replace("'", "")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
    )


# -------------- 导出 Word --------------
@app.post("/export/docx")
def export_docx(
    result: Optional[Dict[str, Any]] = Body(
        None,
        description="可选：完整响应或 customer_meeting_detail；为空或不合法将使用最近一次分析结果",
    ),
    scheme_id: Optional[str] = Query(None),
    meeting_id: Optional[str] = Query(None),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        detail = _try_extract_detail(result) or _get_cached_detail_or_400(creds, scheme_id, meeting_id)
        docx_bytes = export_docx_from_payload(detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"export docx failed: {e}")

    filename = (detail.get("name") or "meeting_minutes").replace('"', "").replace("'", "")
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}.docx"'},
    )
