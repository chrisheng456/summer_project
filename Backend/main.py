# main.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import subprocess
import json
from qdrant_search import smart_search, client, model, collection_name
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
collection_name = "meeting_minutes"

from qdrant_search import smart_search


app = FastAPI(title="AI会议助手 API")

AUDIO_DIR = Path("./uploads")
AUDIO_DIR.mkdir(exist_ok=True)

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """上传音频并处理成会议纪要"""
    try:
        # 保存音频
        audio_path = AUDIO_DIR / "uploaded.wav"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 我在这里先暂定运行 speech_to_text.py、post_process_local.py、qdrant_utils.py 这几个功能 后面根据情况更改
        subprocess.run(["python", "speech_to_text.py"], check=True)
        subprocess.run(["python", "post_process_local.py"], check=True)
        subprocess.run(["python", "qdrant_utils.py"], check=True)

        return {"message": "✅ 上传并处理成功"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/search/")
def search(q: str = Query(..., description="要搜索的问题")):
    """向量语义搜索接口"""
    try:
        query_vector = model.encode(q).tolist()
        results = smart_search(client, collection_name, query_vector)

        return [
            {"text": point.payload.get("text", "[无文本]"), "score": point.score}
            for point in results
        ]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/latest_minutes/")
def latest_minutes():
    """获取最近一次分析完成的 JSON 文件内容"""
    try:
        latest_file = sorted(Path(".").glob("meeting_minutes_*_local.json"))[-1]
        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
