from loguru import logger
from peewee import SqliteDatabase, Model, CharField, DateTimeField, TextField
from .config import DataBaseConfig
import datetime

models = []

db = SqliteDatabase("meeting_minutes.db")


class ConversionTask(Model):
    id = CharField(primary_key=True, max_length=64)
    status = CharField(max_length=32, default="pending")
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)
    result_json = TextField(null=True)
    error_message = TextField(null=True)

    class Meta:
        database = db


models.append(ConversionTask)


def initialize_database(config: DataBaseConfig = None):
    db.connect(reuse_if_open=True)
    db.create_tables(models, safe=True)
    logger.info("Database initialisation complete")
