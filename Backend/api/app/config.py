import os
from dotenv import load_dotenv
from pydantic import Field, BaseModel


# 加载环境变量
load_dotenv(override=True)
# 获取环境变量
env = os.environ.copy()


class DataBaseConfig(BaseModel):
    """
    数据库配置类。 Peewee ORM 使用的配置类。

    通过环境变量配置数据库连接信息。
    """

    # 数据库名称
    db_name: str = Field(
        default="goofish",
        alias="DB_NAME",
    )
    # 数据库用户名
    db_user: str = Field(
        default="root",
        alias="DB_USER",
    )
    # 数据库密码
    db_password: str = Field(
        default="",
        alias="DB_PASS",
    )
    # 数据库主机
    db_host: str = Field(
        default="localhost",
        alias="DB_HOST",
    )
    # 数据库端口
    db_port: int = Field(
        default=3306,
        alias="DB_PORT",
    )


class AzureSpeechConfig(BaseModel):
    """
    Azure 语音服务配置类。

    通过环境变量配置 Azure 语音服务的密钥和区域。
    """

    # Azure 语音服务密钥
    speech_key: str = Field(..., alias="AZURE_SPEECH_KEY")
    # Azure 语音服务区域
    service_region: str = Field(default="ukwest", alias="AZURE_SERVICE_REGION")


# Huggingface Token 配置类
class HuggingfaceConfig(BaseModel):
    """
    Huggingface Token 配置类。

    通过环境变量配置 Huggingface 的访问令牌。
    """

    # Huggingface 访问令牌
    token: str = Field(..., alias="HUGGINGFACE_TOKEN")


class CustomerApiConfig(BaseModel):
    """
    客户API账号密码配置类，通过环境变量配置。
    """

    username: str = Field(..., alias="CUSTOMER_API_USERNAME")
    password: str = Field(..., alias="CUSTOMER_API_PASSWORD")


class Config(BaseModel):

    # 数据库配置
    database: DataBaseConfig = Field(
        default_factory=lambda: DataBaseConfig(**env)
    )

    # Azure 语音服务配置
    azure_speech: AzureSpeechConfig = Field(
        default_factory=lambda: AzureSpeechConfig(**env)
    )

    # Huggingface Token
    huggingface: HuggingfaceConfig = Field(
        default_factory=lambda: HuggingfaceConfig(**env)
    )
    # 客户API账号密码
    customer_api: CustomerApiConfig = Field(
        default_factory=lambda: CustomerApiConfig(**env)
    )


# 创建配置实例
app_config = Config()
