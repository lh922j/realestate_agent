"""
도구별 응답 레이턴시 벤치마크

각 도구를 직접 호출해 응답 시간을 측정합니다.
실행: python -m tests.eval.latency
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

TOOL_CASES = [
    {
        "tool": "query_trade_data",
        "args": {"district": "역삼동", "area_min": 55, "area_max": 65, "year_from": 2024, "year_to": 2026},
    },
    {
        "tool": "query_rent_data",
        "args": {"district": "역삼동", "area_min": 55, "area_max": 65, "rent_type": "전세", "year_from": 2024, "year_to": 2026},
    },
    {
        "tool": "query_trade_nearby",
        "args": {"place_name": "강남역", "radius_km": 1.0, "area_min": 55, "area_max": 65, "year_from": 2024, "year_to": 2026},
    },
    {
        "tool": "predict_price",
        "args": {
            "area_exclusive": 84.0, "floor": 10, "build_year": 2010,
            "district_code": "11680", "latitude": 37.5172, "longitude": 127.0473,
        },
    },
    {
        "tool": "detect_anomaly",
        "args": {"district": "강남구", "area_min": 80, "area_max": 90, "year_from": 2023, "year_to": 2025},
    },
    {
        "tool": "search_area_info",
        "args": {"query": "강남구 학군 정보", "n_results": 3},
    },
]


def run(repeat: int = 3) -> dict:
    from src.agent.tools import get_tools

    tool_map = {t.name: t for t in get_tools()}
    results = {}

    print(f"\n{'='*60}")
    print(f"도구 레이턴시 벤치마크  (각 {repeat}회 평균)")
    print(f"{'='*60}")
    print(f"  {'도구명':<25} {'평균':>8} {'최솟값':>8} {'최댓값':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")

    for tc in TOOL_CASES:
        name = tc["tool"]
        tool = tool_map.get(name)
        if tool is None:
            continue

        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            try:
                tool.invoke(tc["args"])
            except Exception:
                pass
            times.append(time.perf_counter() - t0)

        avg, mn, mx = sum(times) / len(times), min(times), max(times)
        results[name] = {"avg": avg, "min": mn, "max": mx}
        print(f"  {name:<25} {avg:>7.2f}s {mn:>7.2f}s {mx:>7.2f}s")

    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    run()
