#实现对话功能+使用fastapi

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from modules.tools.azure_client import get_azure_client, get_deployment_name


deployment = get_deployment_name()
app = FastAPI()

# 允许 CORS 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 或者 ["http://localhost:5173"] 更安全
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Azure OpenAI 配置

client = get_azure_client()  # 获取 Azure OpenAI 客户端

# 定义请求体的数据模型
class ChatRequest(BaseModel):
    messages: list  # 聊天上下文，支持多轮对话，例如 [{"role": "user", "content": "你好"}]

# 定义响应体的数据模型（用于接口文档规范化，可选）
class ChatResponse(BaseModel):
    reply: str  # AI 返回的回复内容

# 定义 API 路由：POST 请求，地址为 /api/chat
@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(request: ChatRequest):
    """
    与 Azure OpenAI 交互的 API 接口
    请求：接收多轮聊天 messages
    响应：返回 AI 回复的内容
    """
    try:
        # 调用 Azure OpenAI Chat Completions 接口生成回复
        response = client.chat.completions.create(
            model=deployment,           # 指定使用的模型
            messages=request.messages,  # 传递聊天上下文
            max_tokens=1000,            # 限制回复最大长度
            temperature=0.7,            # 控制回答的创造性（值越高越随机）
            top_p=1.0                   # 控制多样性
        )

        # 提取回复内容并去除首尾空格
        reply = response.choices[0].message.content.strip()

        # 返回标准格式的响应
        return {"reply": reply}

    except Exception as e:
        # 如果出现错误，返回错误信息
        return {"reply": f" 出现错误：{str(e)}"}
