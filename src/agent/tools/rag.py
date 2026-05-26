from langchain_core.tools import tool
from loguru import logger

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings
        from ..config import settings
        from ..rag.embeddings import get_embedding_function

        client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name="area_info",
            embedding_function=get_embedding_function(),
        )
    return _collection


@tool
def search_area_info(query: str, n_results: int = 3) -> str:
    """
    지역 정보(학군, 교통, 편의시설, 개발호재 등)를 벡터 검색합니다.

    Args:
        query: 검색 질의, 예: '강남구 학군 정보', '마포구 대중교통'
        n_results: 반환 결과 수 (기본 3)
    """
    logger.info(f"[rag] query='{query}' n_results={n_results}")
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return "지역 정보 DB가 아직 비어있습니다. 데이터 인덱싱이 필요합니다."

        results = collection.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return f"'{query}'에 대한 지역 정보를 찾을 수 없습니다."

        logger.debug(f"[rag] {len(docs)}건 검색됨")
        lines = []
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            area = meta.get("area", "")
            lines.append(f"[{i}] {area}\n{doc}")
        return "\n\n".join(lines)

    except Exception as e:
        logger.error(f"[rag] 오류: {e}")
        return f"검색 오류: {e}"
