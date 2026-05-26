from langchain_core.tools import tool
from loguru import logger
from sqlalchemy import text

from ..db.database import get_engine


@tool
def query_rent_data(
    district: str,
    area_min: float = 0,
    area_max: float = 300,
    rent_type: str = "전체",
    year_from: int = 2024,
    year_to: int = 2026,
    limit: int = 5,
) -> str:
    """
    아파트 전세·월세(임대차) 실거래 내역을 조회합니다.
    매매 거래는 이 툴이 아닌 query_trade_data를 사용해야 합니다.

    Args:
        district: 동명 또는 구명 (예: '역삼동', '마포구')
        area_min: 전용면적 최솟값 (㎡)
        area_max: 전용면적 최댓값 (㎡)
        rent_type: '전세', '월세', '전체' 중 하나
        year_from: 조회 시작 연도
        year_to: 조회 종료 연도
        limit: 반환 건수 (최대 50)
    """
    if not district or not district.strip():
        return "지역 이름이 비어 있습니다. 동 이름이나 구 이름을 알려주세요 (예: 역삼동, 강남구)."
    limit = min(limit, 50)
    logger.info(f"[query_rent] district={district} area={area_min}~{area_max}㎡ type={rent_type} year={year_from}~{year_to}")

    # is_jeonse: 1=전세, 0=월세
    type_filter = ""
    if rent_type == "전세":
        type_filter = "AND is_jeonse = 1"
    elif rent_type == "월세":
        type_filter = "AND is_jeonse = 0"

    sql = text(f"""
        SELECT apt_name, dong_name, area_exclusive, floor,
               deposit, monthly_rent, is_jeonse, deal_date
        FROM apt_rent
        WHERE dong_name LIKE :district
          AND area_exclusive BETWEEN :area_min AND :area_max
          AND deal_year BETWEEN :year_from AND :year_to
          {type_filter}
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

        logger.debug(f"[query_rent] 조회 결과 {len(rows)}건")
        if not rows:
            return f"'{district}' 지역에서 조건에 맞는 전월세 내역이 없습니다."

        lines = ["[ 전세·월세 임대차 실거래 ]"]
        lines.append(f"{'아파트명':<20} {'동명':<10} {'면적':>6} {'유형':>4} {'보증금':>10} {'월세':>8} {'거래일':>12}")
        lines.append("-" * 80)
        for r in rows:
            kind = "전세" if r.is_jeonse else "월세"
            monthly = f"{int(r.monthly_rent):,}만" if not r.is_jeonse else "-"
            lines.append(
                f"{r.apt_name:<20} {r.dong_name:<10} {r.area_exclusive:>5.1f}㎡ "
                f"{kind:>4} {int(r.deposit):>8,}만 {monthly:>8} {str(r.deal_date):>12}"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[query_rent] 오류: {e}")
        return f"조회 오류: {e}"


def fetch_map_points_rent(
    district: str,
    area_min: float = 0,
    area_max: float = 300,
    rent_type: str = "전체",
    year_from: int = 2024,
    year_to: int = 2026,
    limit: int = 50,
) -> list[dict]:
    """Streamlit 지도용 전·월세 좌표 조회."""
    type_filter = ""
    if rent_type == "전세":
        type_filter = "AND r.is_jeonse = 1"
    elif rent_type == "월세":
        type_filter = "AND r.is_jeonse = 0"

    sql = text(f"""
        SELECT r.apt_name, r.dong_name, r.area_exclusive,
               r.deposit, r.monthly_rent, r.is_jeonse,
               g.latitude, g.longitude
        FROM apt_rent r
        JOIN apt_geocode g ON r.apt_name = g.apt_name AND r.dong_name = g.dong_name
        WHERE r.dong_name LIKE :district
          AND r.area_exclusive BETWEEN :area_min AND :area_max
          AND r.deal_year BETWEEN :year_from AND :year_to
          AND g.latitude IS NOT NULL
          {type_filter}
        ORDER BY r.deal_date DESC
        LIMIT :limit
    """)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(sql, {
                "district": f"%{district}%",
                "area_min": area_min, "area_max": area_max,
                "year_from": year_from, "year_to": year_to,
                "limit": limit,
            }).fetchall()
        return [
            {
                "apt_name": r.apt_name, "dong_name": r.dong_name,
                "area_exclusive": r.area_exclusive,
                "deal_amount": float(r.deposit),
                "price_label": f"보증금 {int(r.deposit):,}만" + (f" / 월 {int(r.monthly_rent):,}만" if not r.is_jeonse else ""),
                "rent_type": "전세" if r.is_jeonse else "월세",
                "latitude": r.latitude, "longitude": r.longitude,
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"[fetch_map_points_rent] 오류: {e}")
        return []
