# Backend/api/server.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException,
    Body,
    Response,
    Query,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder

from .app.pipeline import process_pipeline
from .app.utils.pp_client import PPClient
from .app.utils.pdf_exporter import export_to_pdf
from .app.utils.word_exporter import export_to_word

app = FastAPI(title="Meeting Pipeline API")
security = HTTPBearer(auto_error=True)

# ---- 简单内存缓存：按 (token, scheme_id, meeting_id) 保存最近一次的 meeting detail ----
# 也保留一个“最近一次”的总缓存，便于用户只点导出不传任何参数
_LAST_DETAIL_BY_KEY: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_LAST_DETAIL_BY_TOKEN: Dict[str, Dict[str, Any]] = {}


def _safe_len(x: Any) -> int:
    try:
        return len(x) if x is not None else 0
    except Exception:
        return 0


def _extract_detail(payload: Any) -> Dict[str, Any]:
    """
    兼容三种输入：
    1) 直接是 meeting detail 对象（包含 agenda）
    2) 外面包了一层 {"customer_meeting_detail": {...}}
    3) payload 是 str/bytes/bytearray，需要先 json.loads
    """
    if payload is None:
        raise ValueError("empty body")

    if isinstance(payload, (bytes, bytearray, str)):
        payload = json.loads(payload)

    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")

    if "customer_meeting_detail" in payload and isinstance(payload["customer_meeting_detail"], dict):
        return payload["customer_meeting_detail"]

    if "agenda" in payload:
        return payload

    raise ValueError("not a valid meeting detail JSON (missing 'agenda')")


@app.get("/health")
def health():
    return {"ok": True}


# ---------------- 登录 ----------------
@app.post("/auth/login")
def login(
    payload: Dict[str, str] = Body(..., example={"username": "ruixiong", "password": "******"}),
    include_meetings: bool = Query(True),
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


# ---------------- 会议列表/详情 ----------------
@app.get("/customer/meetings")
def list_meetings(creds: HTTPAuthorizationCredentials = Depends(security)):
    token = creds.credentials
    try:
        client = PPClient(bearer_token=token)
        meetings = client.list_meetings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch meetings failed: {e}")
    return {"ok": True, "meetings": meetings}


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


# ---------------- 跑流水线：把结果写入缓存 ----------------
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

    meeting_detail = getattr(info, "customer_meeting_detail", None)
    if isinstance(meeting_detail, dict):
        # 写入两层缓存
        _LAST_DETAIL_BY_KEY[(bearer_token, str(scheme_id), str(meeting_id))] = meeting_detail
        _LAST_DETAIL_BY_TOKEN[bearer_token] = meeting_detail

    stt_lines = getattr(info, "transcription", None)
    cleaned_lines = getattr(info, "cleaned_transcription", None)
    diarization = getattr(info, "speaker_segments", None) or getattr(info, "diarization", None)
    classification = getattr(info, "classified_lines", None)
    label_counts = getattr(info, "label_counts", None)
    summary = getattr(info, "summary", None)

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


# ---------------- 导出：PDF / DOCX ----------------
# 现在 Body 改为可选；如果没给，就用“最近一次分析结果”
def _try_extract_detail(payload: Any) -> Optional[Dict[str, Any]]:
    """
    尝试从 payload 中提取 meeting detail：
    - 直接是 detail（有 agenda）
    - 包了一层 {"customer_meeting_detail": {...}}
    - payload 是 str/bytes 先 json.loads
    解析失败或不合法则返回 None（让调用方回退到缓存）
    """
    if payload is None:
        return None

    try:
        if isinstance(payload, (bytes, bytearray, str)):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            return None

        if "customer_meeting_detail" in payload and isinstance(payload["customer_meeting_detail"], dict):
            detail = payload["customer_meeting_detail"]
        else:
            detail = payload

        if isinstance(detail, dict) and isinstance(detail.get("agenda"), list):
            return detail
        return None
    except Exception:
        return None


def _get_cached_detail_or_400(
    creds,
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
    # 兜底：明确提示需要先跑 analyze
    from fastapi import HTTPException
    raise HTTPException(status_code=400, detail="no cached result found; please run /pipeline/analyze first")


# ---- 替换：/export/pdf ----
@app.post("/export/pdf")
def export_pdf(
    result: Optional[Dict[str, Any]] = Body(None, description="可选：完整响应或 customer_meeting_detail；若为空或不合法则使用最近一次分析结果"),
    scheme_id: Optional[str] = Query(None, description="可选：指定 scheme_id（当同一账号跑过多个会议时）"),
    meeting_id: Optional[str] = Query(None, description="可选：指定 meeting_id（当同一账号跑过多个会议时）"),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        detail = _try_extract_detail(result)
        if detail is None:  # 解析失败/空壳 → 回退缓存
            detail = _get_cached_detail_or_400(creds, scheme_id, meeting_id)

        pdf_bytes = export_to_pdf(detail)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"export pdf failed: {e}")

    filename = (detail.get("name") or "meeting_minutes").replace('"', "").replace("'", "")
    headers = {"Content-Disposition": f'attachment; filename="{filename}.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


# ---- 替换：/export/docx ----
@app.post("/export/docx")
def export_docx(
    result: Optional[Dict[str, Any]] = Body(None, description="可选：完整响应或 customer_meeting_detail；若为空或不合法则使用最近一次分析结果"),
    scheme_id: Optional[str] = Query(None),
    meeting_id: Optional[str] = Query(None),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        detail = _try_extract_detail(result)
        if detail is None:  # 解析失败/空壳 → 回退缓存
            detail = _get_cached_detail_or_400(creds, scheme_id, meeting_id)

        docx_bytes = export_to_word(detail)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"export docx failed: {e}")

    filename = (detail.get("name") or "meeting_minutes").replace('"', "").replace("'", "")
    headers = {"Content-Disposition": f'attachment; filename="{filename}.docx"'}
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )


# ---------------- 一步到位：上传→直接导出（保留） ----------------
@app.post("/pipeline/analyze_export/pdf")
async def analyze_export_pdf(
    scheme_id: str = Query(...),
    meeting_id: str = Query(...),
    file: UploadFile = File(...),
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
        detail = getattr(info, "customer_meeting_detail", None)
        if not detail:
            raise RuntimeError("empty customer_meeting_detail")
        # 写缓存
        _LAST_DETAIL_BY_KEY[(bearer_token, str(scheme_id), str(meeting_id))] = detail
        _LAST_DETAIL_BY_TOKEN[bearer_token] = detail
        pdf_bytes = export_to_pdf(detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze_export pdf failed: {e}")

    filename = (detail.get("name") or "meeting_minutes").replace('"', "").replace("'", "")
    headers = {"Content-Disposition": f'attachment; filename="{filename}.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@app.post("/pipeline/analyze_export/docx")
async def analyze_export_docx(
    scheme_id: str = Query(...),
    meeting_id: str = Query(...),
    file: UploadFile = File(...),
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
        detail = getattr(info, "customer_meeting_detail", None)
        if not detail:
            raise RuntimeError("empty customer_meeting_detail")
        _LAST_DETAIL_BY_KEY[(bearer_token, str(scheme_id), str(meeting_id))] = detail
        _LAST_DETAIL_BY_TOKEN[bearer_token] = detail
        docx_bytes = export_to_word(detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze_export docx failed: {e}")

    filename = (detail.get("name") or "meeting_minutes").replace('"', "").replace("'", "")
    headers = {"Content-Disposition": f'attachment; filename="{filename}.docx"'}
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )
