# Backend/api/server.py
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .app.pipeline import process_pipeline
from .app.utils.pp_client import PPClient

app = FastAPI(title="Meeting Pipeline API")
# 声明 Bearer 认证方案；只要任一接口使用了它，Swagger 右上角就会出现 Authorize 按钮
security = HTTPBearer(auto_error=True)


def _safe_len(x: Any) -> int:
    try:
        return len(x) if x is not None else 0
    except Exception:
        return 0


@app.get("/health")
def health():
    return {"ok": True}


# ---------------- 登录：返回真正的 bearer_token（JWT） ----------------
@app.post("/auth/login")
def login(payload: Dict[str, str] = Body(...)):
    """
    简单本地校验 + 远端登录：
    - 本地仅允许 {ruixiong / Ruixiong24937!}
    - 成功后调用客户 API /Logon，返回远端的 bearer_token（JWT）
    - 这个 token 就是你要在 Swagger 右上角 Authorize 里粘贴的“纯 token”（不要写 Bearer ）
    """
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()

    if not (username == "ruixiong" and password == "Ruixiong24937!"):
        raise HTTPException(status_code=401, detail="invalid credentials")

    try:
        client = PPClient()
        bearer = client.login(username, password)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"remote login failed: {e}")

    return {"ok": True, "token": bearer}


# ---------------- 拉会议列表（需要 Authorize） ----------------
@app.get("/customer/meetings")
def list_meetings(creds: HTTPAuthorizationCredentials = Depends(security)):
    """
    使用 Authorize 配置的 Bearer token（粘贴纯 JWT）拉取会议列表。
    """
    token = creds.credentials  # 纯 token
    try:
        client = PPClient(bearer_token=token)
        meetings = client.list_meetings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch meetings failed: {e}")
    return {"ok": True, "meetings": meetings}


# ---------------- 跑整条流水线（需要 Authorize） ----------------
@app.post("/pipeline/analyze")
async def analyze(
    scheme_id: str,
    meeting_id: str,
    file: UploadFile = File(...),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    """
    前端上传音频 + 选择 scheme / meeting 后调用。
    Authorize 里填的 Bearer token 会传入流水线，
    用于在 S00 客户 API 步获取 meeting detail。
    """
    bearer_token = creds.credentials
    data = await file.read()

    try:
        info = process_pipeline(
            input_file_content=data,
            scheme_id=scheme_id,
            meeting_id=meeting_id,
            bearer_token=bearer_token,  # <== 传下去给 S00 使用客户 API
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")

    # 组织统一响应
    stt_lines = getattr(info, "transcription", None)                # S01
    cleaned_lines = getattr(info, "cleaned_transcription", None)    # S02
    diarization = getattr(info, "speaker_segments", None) or getattr(info, "diarization", None)  # S03
    classification = getattr(info, "classified_lines", None)        # S05
    label_counts = getattr(info, "label_counts", None)              # S05
    summary = getattr(info, "summary", None)                        # S06
    meeting_detail = getattr(info, "customer_meeting_detail", None) # S00

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
