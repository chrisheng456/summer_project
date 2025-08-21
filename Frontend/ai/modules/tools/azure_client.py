from openai import AzureOpenAI
from .get_env import load_azure_openai_config  

# Connect to Azure OpenAI API
def get_azure_client():
    """
    Get Azure OpenAI client
    """
    config = load_azure_openai_config()
    client = AzureOpenAI(
        api_key=config["api_key"],
        azure_endpoint=config["endpoint"],
        api_version=config["api_version"],
    )
    return client

# Get deployment (model) name
def get_deployment_name():
    config = load_azure_openai_config()
    return config["deployment"]