# Backend/api/server.py
from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Optional, List

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

def _finalize_minutes(detail: Dict[str, Any]) -> Dict[str, Any]:
    """
    产出“最终版会议纪要”：
    - agenda 仅保留基础元数据 + 我们生成的 label/label_score/explanation/summary
    - 同时保留最终分配后的 lines（只留 start/end/text/speaker 四项）
    """
    if not isinstance(detail, dict):
        return {}

    keep_top = detail.copy()
    keep_top.pop("agenda", None)

    out_agenda: List[Dict[str, Any]] = []
    for it in (detail.get("agenda") or []):
        if not isinstance(it, dict):
            continue

        # 1) 基础字段
        slim = {}
        for k in [
            "id", "number", "title", "indent", "owner", "action",
            "calculatedStartTime", "lengthMinutes", "action_colour",
        ]:
            if k in it:
                slim[k] = it[k]

        # 2) 我们生成的字段
        for k in ["label", "label_score", "explanation", "summary"]:
            if k in it:
                slim[k] = it[k]

        # 3) 最终版 lines（精简字段）
        kept_lines = []
        for ln in it.get("lines", []) or []:
            kept_lines.append({
                "start": float(ln.get("start")) if ln.get("start") is not None else None,
                "end":   float(ln.get("end"))   if ln.get("end")   is not None else None,
                "text":  (ln.get("text") or "").strip(),
                "speaker": ln.get("speaker", "Unknown"),
            })
        kept_lines.sort(key=lambda x: (x["start"] if x["start"] is not None else float("inf"),
                                       x["end"] if x["end"] is not None else float("inf")))
        slim["lines"] = kept_lines

        out_agenda.append(slim)

    keep_top["agenda"] = out_agenda
    return keep_top

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
# --- 完全替换原 /pipeline/analyze 路由 ---
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
    if not isinstance(meeting_detail, dict):
        raise HTTPException(status_code=500, detail="no meeting detail produced")

    # 仅保留最终版
    final_detail = _finalize_minutes(meeting_detail)

    # 缓存里也只存最终版，供 /export/pdf|docx 使用
    token = bearer_token
    _LAST_DETAIL_BY_TOKEN[token] = final_detail
    _LAST_DETAIL_BY_KEY[(token, str(scheme_id), str(meeting_id))] = final_detail

    return jsonable_encoder({
        "ok": True,
        "final_minutes": final_detail
    })


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
