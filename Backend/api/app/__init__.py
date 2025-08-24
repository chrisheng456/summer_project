from fastapi import FastAPI


from .config import app_config
from .api_legacy import router
from .models import initialize_database

# Creating a FastAPI application instance
app = FastAPI()

# Includes API routing
app.include_router(router, tags=["api"])


async def startup_db() -> None:
    """
    Start database connection
    """
    initialize_database(app_config.database)

app.on_event("startup")(startup_db)
