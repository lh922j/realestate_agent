import sys
from pathlib import Path
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parents[4]
_DEFAULT_DB = str(_PROJECT_ROOT / "realestate" / "data" / "processed" / "realestate.db")
_DEFAULT_MODEL = str(_PROJECT_ROOT / "realestate" / "data" / "models" / "price_model_trade_lgbm_complex.pkl")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    kakao_api_key: str = ""
    database_url: str = f"sqlite:///{_DEFAULT_DB}"
    model_path: str = _DEFAULT_MODEL
    chroma_path: str = "./data/chroma"
    log_level: str = "INFO"


settings = Settings()

# 로그 설정: 콘솔 + 파일 동시 출력
_LOG_DIR = Path(__file__).parents[3] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level=settings.log_level,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add(_LOG_DIR / "debug.log", level="DEBUG",
           rotation="10 MB", retention="7 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}")
