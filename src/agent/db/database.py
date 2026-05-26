from functools import lru_cache
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from ..config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = settings.database_url
    logger.info(f"[db] 연결: {url}")
    engine = create_engine(url, echo=False)
    return engine


def get_session() -> Session:
    SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return SessionLocal()
