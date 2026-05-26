"""
BGE-M3 임베딩 함수 (ChromaDB용)

BAAI/bge-m3: 한국어 포함 100개 언어 지원 다국어 임베딩 모델
- 로컬 실행으로 API 비용 없음
- 첫 실행 시 HuggingFace에서 자동 다운로드 (~2GB)
"""

from functools import lru_cache
from loguru import logger


class BGEEmbeddingFunction:
    """ChromaDB EmbeddingFunction 인터페이스 구현."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[embedding] {self.model_name} 로드 중 (최초 1회)...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("[embedding] 모델 로드 완료")
        return self._model

    def name(self) -> str:
        return self.model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = self._load()
        embeddings = model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, input=None, **kwargs) -> list[list[float]]:
        if isinstance(input, list):
            return self.__call__(input)
        return self.__call__([input])

    def embed_documents(self, input=None, **kwargs) -> list[list[float]]:
        return self.__call__(input or [])


@lru_cache(maxsize=1)
def get_embedding_function() -> BGEEmbeddingFunction:
    return BGEEmbeddingFunction()
