from fastapi import FastAPI


from .config import app_config
from .api_legacy import router
from .models import initialize_database

# 创建 FastAPI 应用程序实例
app = FastAPI()
# 包含 API 路由
app.include_router(router, tags=["api"])


async def startup_db() -> None:
    """
    启动数据库连接
    """
    initialize_database(app_config.database)


app.on_event("startup")(startup_db)
