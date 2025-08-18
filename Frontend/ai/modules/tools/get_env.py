from dotenv import load_dotenv
import os

def load_azure_openai_config():
    """
    加载 Azure OpenAI 的所有配置，从 .env 或环境变量
    """
    load_dotenv()
    return {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    }