from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from modules.tools.azure_client import get_azure_client, get_deployment_name


deployment = get_deployment_name()
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



client = get_azure_client()  # Get Azure OpenAI client

# Define request body data model
class ChatRequest(BaseModel):
    messages: list  

# Define response body data model 
class ChatResponse(BaseModel):
    reply: str 

# Define API route: POST request, endpoint /api/chat
@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(request: ChatRequest):
    """
    API endpoint to interact with Azure OpenAI
    Request: receives multi-turn conversation messages
    Response: returns AI-generated reply
    """
    try:
        # Call Azure OpenAI Chat Completions API
        response = client.chat.completions.create(
            model=deployment,          
            messages=request.messages,  
            max_tokens=1000,            
            temperature=0.7,            
            top_p=1.0                   
        )

        # Extract reply content and remove whitespace
        reply = response.choices[0].message.content.strip()

        # Return standardized response format
        return {"reply": reply}

    except Exception as e:
        # Return error message if something goes wrong
        return {"reply": f" Error occurred：{str(e)}"}
