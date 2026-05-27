"""시군구 이름 → sgg_code 매핑 헬퍼.

매핑 데이터는 src/agent/rag/district_codes.json에서 로드합니다.
새 지역 추가 시 Python 코드가 아닌 JSON 파일만 수정하면 됩니다.
"""

import json
from functools import lru_cache
from pathlib import Path

_CODES_PATH = Path(__file__).parents[1] / "rag" / "district_codes.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, list[str]]:
    """JSON에서 {구이름: [sgg_code, ...]} 평탄화 딕셔너리 반환."""
    with open(_CODES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    flat: dict[str, list[str]] = {}
    for region_codes in data.values():
        for name, codes in region_codes.items():
            # 인천 중구/동구/서구 등 이름 중복 시 코드 합산
            flat.setdefault(name, [])
            for c in codes:
                if c not in flat[name]:
                    flat[name].append(c)
    return flat


def get_sgg_codes(district: str) -> list[str]:
    """구/시 이름으로 sgg_code 목록 반환. 매핑 없으면 빈 리스트."""
    return _load().get(district.strip(), [])


def district_sql_filter(district: str, dong_col: str = "dong_name", sgg_col: str = "sgg_code") -> tuple[str, dict]:
    """
    dong_name 또는 sgg_code 기반 WHERE 조건과 파라미터 반환.

    - dong 이름(역삼동 등): dong_name LIKE '%역삼동%'
    - 구/시 이름(강남구 등): dong_name LIKE '%강남구%' OR sgg_code IN ('11680')
    """
    codes = get_sgg_codes(district)
    if codes:
        placeholders = ",".join(f"'{c}'" for c in codes)
        sql = f"({dong_col} LIKE :district OR {sgg_col} IN ({placeholders}))"
    else:
        sql = f"{dong_col} LIKE :district"
    return sql, {"district": f"%{district}%"}
