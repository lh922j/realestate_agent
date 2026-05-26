"""
지식 베이스 구축 스크립트

실행: python -m src.agent.rag.indexer

두 가지 데이터를 ChromaDB에 인덱싱합니다:
1. DB 집계 → 동별 시세 요약 (자동 생성)
2. 지역 특성 → 교통·생활 정보 (area_info.json)
"""

import json
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from ..config import settings
from ..db.database import get_engine

_AREA_INFO_PATH = Path(__file__).parent / "area_info.json"


def _get_chroma_collection():
    import chromadb
    from chromadb.config import Settings
    from .embeddings import get_embedding_function

    client = chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name="area_info",
        embedding_function=get_embedding_function(),
    )


def build_price_summaries() -> list[dict]:
    """DB에서 동별 면적대 시세 요약 문서를 생성합니다."""
    sql = text("""
        SELECT
            dong_name,
            COUNT(*) as total_cnt,
            ROUND(AVG(deal_amount), 0) as avg_price,
            ROUND(MIN(deal_amount), 0) as min_price,
            ROUND(MAX(deal_amount), 0) as max_price,
            ROUND(AVG(CASE WHEN area_exclusive < 60 THEN deal_amount END), 0) as avg_small,
            ROUND(AVG(CASE WHEN area_exclusive BETWEEN 60 AND 85 THEN deal_amount END), 0) as avg_mid,
            ROUND(AVG(CASE WHEN area_exclusive > 85 THEN deal_amount END), 0) as avg_large,
            MAX(deal_date) as latest_date
        FROM apt_trade
        WHERE deal_year >= 2023
        GROUP BY dong_name
        HAVING total_cnt >= 10
        ORDER BY total_cnt DESC
    """)

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    docs = []
    for r in rows:
        parts = [f"{r.dong_name} 아파트 실거래가 시세 요약 (2023년 이후 {r.total_cnt:,}건 기준)"]
        parts.append(f"전체 평균: {int(r.avg_price):,}만원 | 최저: {int(r.min_price):,}만원 | 최고: {int(r.max_price):,}만원")

        if r.avg_small:
            parts.append(f"소형(~59㎡) 평균: {int(r.avg_small):,}만원")
        if r.avg_mid:
            parts.append(f"중형(60~85㎡) 평균: {int(r.avg_mid):,}만원")
        if r.avg_large:
            parts.append(f"대형(85㎡~) 평균: {int(r.avg_large):,}만원")

        parts.append(f"최근 거래일: {r.latest_date}")

        docs.append({
            "id": f"price_{r.dong_name}",
            "text": "\n".join(parts),
            "metadata": {"area": r.dong_name, "type": "price_summary"},
        })

    logger.info(f"[indexer] 시세 요약 {len(docs)}건 생성")
    return docs


def build_area_info_docs() -> list[dict]:
    """area_info.json에서 지역 특성 문서를 로드합니다."""
    if not _AREA_INFO_PATH.exists():
        logger.warning(f"[indexer] {_AREA_INFO_PATH} 없음 → 지역 특성 스킵")
        return []

    with open(_AREA_INFO_PATH, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for area_name, info in data.items():
        text_parts = [f"{area_name} 지역 특성 정보"]
        for category, content in info.items():
            text_parts.append(f"[{category}] {content}")

        docs.append({
            "id": f"area_{area_name}",
            "text": "\n".join(text_parts),
            "metadata": {"area": area_name, "type": "area_info"},
        })

    logger.info(f"[indexer] 지역 특성 {len(docs)}건 로드")
    return docs


def run_indexing(reset: bool = False):
    """전체 인덱싱 실행."""
    collection = _get_chroma_collection()

    if reset:
        existing = collection.count()
        if existing > 0:
            collection.delete(where={"type": {"$in": ["price_summary", "area_info"]}})
            logger.info(f"[indexer] 기존 {existing}건 삭제")

    all_docs = build_price_summaries() + build_area_info_docs()

    if not all_docs:
        logger.warning("[indexer] 인덱싱할 문서가 없습니다")
        return

    # 배치 단위로 upsert (OpenAI API 속도 제한 고려)
    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        collection.upsert(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        logger.info(f"[indexer] {i + len(batch)}/{len(all_docs)} 완료")

    logger.info(f"[indexer] 인덱싱 완료 → 총 {collection.count()}건")


if __name__ == "__main__":
    run_indexing(reset=True)
