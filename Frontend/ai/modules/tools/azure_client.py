from openai import AzureOpenAI
from .get_env import load_azure_openai_config  # 绝对/相对都行

#连接api
def get_azure_client():
    """
    获取 Azure OpenAI 客户端
    """
    config = load_azure_openai_config()
    client = AzureOpenAI(
        api_key=config["api_key"],
        azure_endpoint=config["endpoint"],
        api_version=config["api_version"],
    )
    return client

#获取模型名的
def get_deployment_name():
    config = load_azure_openai_config()
    return config["deployment"]