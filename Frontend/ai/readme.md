# ai_content_edit 说明文档

本模块用于通过 **Azure OpenAI** 实现页面内容的智能编辑/改写，提供 API 服务接口（基于 FastAPI 实现），支持灵活的环境变量配置与模块化开发，方便在不同项目中集成。

---

## 目录结构

```text
PythonProject/
│
├── env/
│   └── .env                    # 环境变量配置文件
│
├── modules/
│   └── ai_content_edit/
│       ├── __init__.py         # 包声明文件
│       ├── chat_api.py         # FastAPI 服务接口（API 入口）
│       └── __init__.py         # 包初始化文件（如无特殊需求可保留一个）
│
├── tools/
│   ├── azure_client.py         # Azure OpenAI 客户端封装（负责连接 OpenAI）
│   └── get_env.py              # 环境变量读取工具函数
│
├── readme.md                   # 说明文档
├── requirements.txt            # Python 依赖列表
└── venv/                       # 虚拟环境（自动生成）
```
## 文件说明
```
    •	env/.env
环境变量配置文件，存放敏感配置（如 API KEY、endpoint 等）。不要上传到 Git 仓库。
	•	modules/ai_content_edit/azure_client.py
Azure OpenAI 客户端封装，从环境变量获取密钥和配置。（工具）
	•	modules/ai_content_edit/chat_api.py
FastAPI 主服务，提供 /api/chat POST 接口，实现与 Azure OpenAI 对话 API 对接。（模块功能）
	•	modules/ai_content_edit/tools/get_env.py（工具）
通用环境变量读取工具（如需复用环境变量读取逻辑）。
	•	modules/ai_content_edit/__init__.py（包声明）
包声明文件，可做初始化导入，也可为空。
	•	requirements.txt（依赖）
所有依赖的 Python 库，例如 fastapi、uvicorn、openai、python-dotenv 等。
```

## 使用说明

### 1.	安装依赖
 `pip install -r requirements.txt`
 
### 2.	配置 .env
`在 env/.env 文件中写入（不要上传到代码仓库,可以在venv/.gitignore中添加.env
这样就不会上传到git上）：`

``` 
AZURE_OPENAI_API_KEY=你的API密钥
AZURE_OPENAI_ENDPOINT=https://xxxxxx.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o

后续合并其他模型
```
### 3.	启动服务
`进入 Python 项目根目录(或者chat_api界面输入uvicorn chat_api:app --reload)`

`根目录：uvicorn modules.ai_content_edit.chat_api:app --reload`

### 4.	API 示例
	•	POST /api/chat
	•	请求体格式： "messages": [
    {"role": "system", "content": "你是内容编辑助手"},
    {"role": "user", "content": "请帮我总结一下下面的内容..."}] 

    •	响应体格式:{"reply": "这里是AI的回复"}
  

