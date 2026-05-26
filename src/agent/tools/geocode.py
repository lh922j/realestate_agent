import httpx
from langchain_core.tools import tool
from loguru import logger

from ..config import settings

_KAKAO_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


@tool
def get_station_coordinates(place_name: str) -> str:
    """
    장소 이름(역, 랜드마크, 공원 등)으로 정확한 위도·경도를 조회합니다.
    역 이름이 포함된 질문에는 반드시 이 도구를 먼저 호출한 뒤
    query_trade_nearby 또는 query_rent_nearby에 좌표를 전달하세요.
    절대로 좌표를 직접 추측하지 마세요.

    Args:
        place_name: 장소명 (예: '작전역', '센트럴파크역', '잠실 롯데타워')
    """
    if not settings.kakao_api_key:
        return "KAKAO_API_KEY가 설정되지 않았습니다. .env 파일에 KAKAO_API_KEY를 추가해주세요."

    logger.info(f"[geocode] 장소 검색: {place_name}")
    try:
        resp = httpx.get(
            _KAKAO_URL,
            params={"query": place_name, "size": 1},
            headers={"Authorization": f"KakaoAK {settings.kakao_api_key}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        docs = resp.json().get("documents", [])

        if not docs:
            return f"'{place_name}'의 위치를 찾을 수 없습니다. 정확한 이름으로 다시 시도해주세요."

        doc = docs[0]
        lat = float(doc["y"])
        lon = float(doc["x"])
        name = doc.get("place_name", place_name)
        address = doc.get("road_address_name") or doc.get("address_name", "")
        logger.debug(f"[geocode] {name} → lat={lat}, lon={lon}")
        return f"{name} 좌표: latitude={lat}, longitude={lon} ({address})"

    except Exception as e:
        logger.error(f"[geocode] 오류: {e}")
        return f"좌표 조회 오류: {e}"
