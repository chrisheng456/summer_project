# Backend/api/server.py
import uuid
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .app.utils.pp_client import PPClient
from .app.pipeline import process_pipeline
from .app.pipeline.s04_customer_api import CustomerApiPipeline
# --- force tmp on D: ---
import os, tempfile, pathlib

WORK_TMP = r"D:\summer_project\tmp"            # 按需改成你的 D 盘目录
pathlib.Path(WORK_TMP).mkdir(parents=True, exist_ok=True)

os.environ["TMP"]  = WORK_TMP                  # Windows 下 tempfile 识别 TMP/TEMP
os.environ["TEMP"] = WORK_TMP
tempfile.tempdir   = WORK_TMP                  # 进一步显式指定

print("TEMP DIR =>", tempfile.gettempdir())    # 可见到 D:\summer_project\tmp
# --- end ---


app = FastAPI(title="Meeting Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ===== 会话与 Bearer 安全方案 =====
SESSIONS: Dict[str, Dict[str, Any]] = {}
security = HTTPBearer(bearerFormat="Bearer", auto_error=False)

def get_session(creds: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """从 Bearer Token 里取出会话；供需要鉴权的接口复用。"""
    if not creds or (creds.scheme or "").lower() != "bearer":
        raise HTTPException(status_code=401, detail="missing Authorization header")
    token = creds.credentials
    sess = SESSIONS.get(token)
    if not sess:
        raise HTTPException(status_code=401, detail="invalid token")
    return sess

# ====== 登录，换取 token，同时拉取会议列表 ======
@app.post("/auth/login")
async def login(payload: Dict[str, str]):
    username = payload.get("username")
    password = payload.get("password")

    # demo：仍支持硬编码；否则走真实登录
    if username == "ruixiong" and password == "Ruixiong24937!":
        bearer = PPClient.login(username, password)
    else:
        raise HTTPException(status_code=401, detail="invalid credentials")

    meetings = CustomerApiPipeline.list_meetings(bearer)
    token = str(uuid.uuid4())
    SESSIONS[token] = {"user": username, "bearer": bearer, "meetings": meetings}
    return {"ok": True, "token": token, "meetings": meetings}

# ====== 需要鉴权的接口（自动在 Swagger 显示 Authorize 按钮） ======
@app.get("/customer/meetings")
def list_meetings(sess: Dict[str, Any] = Depends(get_session)):
    return {"ok": True, "meetings": sess["meetings"]}

@app.post("/pipeline/analyze")
async def analyze(
    scheme_id: str,
    meeting_id: str,
    file: UploadFile = File(...),
    sess: Dict[str, Any] = Depends(get_session),
):
    bearer = sess["bearer"]
    content = await file.read()

    info = process_pipeline(
        input_file_content=content,
        scheme_id=scheme_id,
        meeting_id=meeting_id,
        bearer_token=bearer,
    )
    return {
        "ok": True,
        "customer_meeting_detail": getattr(info, "customer_meeting_detail", None),
        "result": getattr(info, "result", None),
    }

# （可选）健康检查
@app.get("/health")
def health():
    return {"status": "ok"}
