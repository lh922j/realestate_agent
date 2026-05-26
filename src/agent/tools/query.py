from langchain_core.tools import tool
from loguru import logger
from sqlalchemy import text

from ..db.database import get_engine


@tool
def query_trade_data(
    district: str,
    area_min: float = 0,
    area_max: float = 300,
    year_from: int = 2022,
    year_to: int = 2026,
    limit: int = 5,
) -> str:
    """
    아파트 매매(매수·매도) 실거래 내역을 조회합니다.
    전세·월세는 이 툴이 아닌 query_rent_data를 사용해야 합니다.

    Args:
        district: 동명 또는 구명 (예: '강남구', '역삼동', '마포')
        area_min: 전용면적 최솟값 (㎡)
        area_max: 전용면적 최댓값 (㎡)
        year_from: 조회 시작 연도
        year_to: 조회 종료 연도
        limit: 반환 건수 (최대 50)
    """
    if not district or not district.strip():
        return "지역 이름이 비어 있습니다. 동 이름이나 구 이름을 알려주세요 (예: 역삼동, 강남구)."
    limit = min(limit, 50)
    logger.info(f"[query] district={district} area={area_min}~{area_max}㎡ year={year_from}~{year_to} limit={limit}")

    sql = text("""
        SELECT apt_name, dong_name, area_exclusive, floor, deal_amount, deal_date
        FROM apt_trade
        WHERE dong_name LIKE :district
          AND area_exclusive BETWEEN :area_min AND :area_max
          AND deal_year BETWEEN :year_from AND :year_to
        ORDER BY deal_date DESC
        LIMIT :limit
    """)

    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(sql, {
                "district": f"%{district}%",
                "area_min": area_min,
                "area_max": area_max,
                "year_from": year_from,
                "year_to": year_to,
                "limit": limit,
            }).fetchall()

        logger.debug(f"[query] 조회 결과 {len(rows)}건")
        if not rows:
            return f"'{district}' 지역에서 조건에 맞는 거래 내역이 없습니다."

        lines = ["[ 매매 실거래 ]"]
        lines.append(f"{'아파트명':<20} {'동명':<10} {'면적':>6} {'층':>4} {'매매금액':>12} {'거래일':>12}")
        lines.append("-" * 78)
        for r in rows:
            lines.append(
                f"{r.apt_name:<20} {r.dong_name:<10} {r.area_exclusive:>5.1f}㎡ "
                f"{r.floor:>3}층 {int(r.deal_amount):>10,}만원 {str(r.deal_date):>12}"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[query] 오류: {e}")
        return f"조회 오류: {e}"


def fetch_map_points(
    district: str,
    area_min: float = 0,
    area_max: float = 300,
    year_from: int = 2022,
    year_to: int = 2026,
    limit: int = 50,
) -> list[dict]:
    """Streamlit에서 직접 호출하는 좌표 조회 함수. @tool 밖에 있어서 config 의존 없음."""
    sql = text("""
        SELECT t.apt_name, t.dong_name, t.area_exclusive, t.deal_amount,
               g.latitude, g.longitude
        FROM apt_trade t
        LEFT JOIN apt_geocode g ON t.apt_name = g.apt_name AND t.dong_name = g.dong_name
        WHERE t.dong_name LIKE :district
          AND t.area_exclusive BETWEEN :area_min AND :area_max
          AND t.deal_year BETWEEN :year_from AND :year_to
          AND g.latitude IS NOT NULL
        ORDER BY t.deal_date DESC
        LIMIT :limit
    """)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(sql, {
                "district": f"%{district}%",
                "area_min": area_min,
                "area_max": area_max,
                "year_from": year_from,
                "year_to": year_to,
                "limit": limit,
            }).fetchall()
        return [
            {
                "apt_name": r.apt_name,
                "dong_name": r.dong_name,
                "area_exclusive": r.area_exclusive,
                "deal_amount": float(r.deal_amount),
                "latitude": r.latitude,
                "longitude": r.longitude,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"[fetch_map_points] 오류: {e}")
        return []
