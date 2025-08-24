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

# ---- Simple in-memory cache: store the last meeting detail per (token, scheme_id, meeting_id)
# Also keep a global "last detail per token" cache, useful for export endpoints without parameters
_LAST_DETAIL_BY_KEY: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_LAST_DETAIL_BY_TOKEN: Dict[str, Dict[str, Any]] = {}


def _safe_len(x: Any) -> int:
    """Safely compute len(x), fallback to 0 if invalid."""
    try:
        return len(x) if x is not None else 0
    except Exception:
        return 0


def _extract_detail(payload: Any) -> Dict[str, Any]:
    """
    Extract meeting detail from different input formats:

    1) Directly a meeting detail object (with "agenda")
    2) Wrapped in {"customer_meeting_detail": {...}}
    3) A str/bytes/bytearray payload → parsed with json.loads
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
    """Health check endpoint."""
    return {"ok": True}


# ---------------- Authentication ----------------
@app.post("/auth/login")
def login(
    payload: Dict[str, str] = Body(..., example={"username": "ruixiong", "password": "******"}),
    include_meetings: bool = Query(True),
):
    """
    Authenticate with the external PPClient service.
    Optionally also fetch the meeting list after login.
    """
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


# ---------------- Meeting list / details ----------------
@app.get("/customer/meetings")
def list_meetings(creds: HTTPAuthorizationCredentials = Depends(security)):
    """Fetch meeting list using the given token."""
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
    """Fetch meeting detail by scheme_id and meeting_id."""
    token = creds.credentials
    try:
        client = PPClient(bearer_token=token)
        detail = client.meeting_detail(scheme_id, meeting_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch meeting detail failed: {e}")
    return {"ok": True, "detail": detail}


# ---------------- Run full pipeline and store result in cache ----------------
@app.post("/pipeline/analyze")
async def analyze(
    scheme_id: str = Query(..., description="Scheme ID"),
    meeting_id: str = Query(..., description="Meeting ID"),
    file: UploadFile = File(..., description="Raw meeting audio file"),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Run the full meeting pipeline:
      - Transcribe and diarize audio
      - Clean text
      - Align text with agenda
      - Classify agenda items
      - Generate summaries
    Results are cached for later export.
    """
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
        # Write result to cache (both keyed and per-token)
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


# ---------------- Export endpoints: PDF / DOCX ----------------
def _try_extract_detail(payload: Any) -> Optional[Dict[str, Any]]:
    """
    Try to extract meeting detail from a request body:
      - Directly a detail dict (with agenda)
      - Wrapped inside {"customer_meeting_detail": {...}}
      - str/bytes payload parsed with json.loads

    Returns None if parsing fails (caller should fallback to cache).
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
    """Fetch detail from cache, or raise 400 if not found."""
    token = creds.credentials
    if scheme_id and meeting_id:
        key = (token, str(scheme_id), str(meeting_id))
        if key in _LAST_DETAIL_BY_KEY:
            return _LAST_DETAIL_BY_KEY[key]
    if token in _LAST_DETAIL_BY_TOKEN:
        return _LAST_DETAIL_BY_TOKEN[token]

    from fastapi import HTTPException
    raise HTTPException(status_code=400, detail="no cached result found; please run /pipeline/analyze first")


@app.post("/export/pdf")
def export_pdf(
    result: Optional[Dict[str, Any]] = Body(None, description="Optional: full pipeline response or customer_meeting_detail. If missing or invalid, fallback to cached result."),
    scheme_id: Optional[str] = Query(None, description="Optional: specify scheme_id if multiple meetings analyzed under the same account"),
    meeting_id: Optional[str] = Query(None, description="Optional: specify meeting_id if multiple meetings analyzed under the same account"),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    """Export the analyzed meeting result as PDF."""
    try:
        detail = _try_extract_detail(result)
        if detail is None:
            detail = _get_cached_detail_or_400(creds, scheme_id, meeting_id)

        pdf_bytes = export_to_pdf(detail)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"export pdf failed: {e}")

    filename = (detail.get("name") or "meeting_minutes").replace('"', "").replace("'", "")
    headers = {"Content-Disposition": f'attachment; filename="{filename}.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@app.post("/export/docx")
def export_docx(
    result: Optional[Dict[str, Any]] = Body(None, description="Optional: full pipeline response or customer_meeting_detail. If missing or invalid, fallback to cached result."),
    scheme_id: Optional[str] = Query(None),
    meeting_id: Optional[str] = Query(None),
    creds: HTTPAuthorizationCredentials = Depends(security),
):
    """Export the analyzed meeting result as Word (.docx)."""
    try:
        detail = _try_extract_detail(result)
        if detail is None:
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
