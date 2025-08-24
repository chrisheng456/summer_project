import os
from dotenv import load_dotenv
from pydantic import Field, BaseModel

# Load environment variables from a .env file if present
load_dotenv(override=True)

# Copy environment variables into a local dict
env = os.environ.copy()


class DataBaseConfig(BaseModel):
    """
    Database configuration class for Peewee ORM.

    Values are loaded from environment variables.
    """

    db_name: str = Field(
        default="goofish",
        alias="DB_NAME",
    )
    db_user: str = Field(
        default="root",
        alias="DB_USER",
    )
    db_password: str = Field(
        default="",
        alias="DB_PASS",
    )
    db_host: str = Field(
        default="localhost",
        alias="DB_HOST",
    )
    db_port: int = Field(
        default=3306,
        alias="DB_PORT",
    )


class AzureSpeechConfig(BaseModel):
    """
    Azure Speech service configuration.
    """
    speech_key: str = Field(..., alias="AZURE_SPEECH_KEY")
    service_region: str = Field(default="uksouth", alias="AZURE_SPEECH_REGION")


class HuggingfaceConfig(BaseModel):
    """
    Hugging Face access token configuration.

    The token is loaded from environment variables.
    """
    token: str = Field(..., alias="HUGGINGFACE_TOKEN")


class CustomerApiConfig(BaseModel):
    """
    Customer API configuration.
    Includes credentials for Azure Speech and Storage.
    """
    speech_key: str = Field(..., alias="AZURE_SPEECH_KEY")
    service_region: str = Field(default="uksouth", alias="AZURE_SPEECH_REGION")
    storage_connection_string: str = Field(..., alias="AZURE_STORAGE_CONNECTION_STRING")
    storage_container: str = Field(..., alias="AZURE_STORAGE_CONTAINER")


class Config(BaseModel):
    """
    Main application configuration wrapper.
    Combines all individual config sections into a single object.
    """

    database: DataBaseConfig = Field(
        default_factory=lambda: DataBaseConfig(**env)
    )

    azure_speech: AzureSpeechConfig = Field(
        default_factory=lambda: AzureSpeechConfig(**env)
    )

    huggingface: HuggingfaceConfig = Field(
        default_factory=lambda: HuggingfaceConfig(**env)
    )

    customer_api: CustomerApiConfig = Field(
        default_factory=lambda: CustomerApiConfig(**env)
    )


# Create a global configuration instance
app_config = Config()
