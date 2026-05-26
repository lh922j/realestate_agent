"""
도구 선택 정확도 평가

에이전트가 질문에 맞는 도구를 올바르게 선택하는지 측정합니다.
실행: python -m tests.eval.tool_selection
"""

from langchain_core.messages import HumanMessage

TEST_CASES = [
    # ── 매매 (행정구역) ──────────────────────────────────────────────
    {"question": "역삼동 59㎡ 매매 최근 거래 알려줘",          "expected": "query_trade_data"},
    {"question": "강남구 84㎡ 아파트 매매 내역 보여줘",         "expected": "query_trade_data"},
    {"question": "마포구 59㎡ 매매 시세 알려줘",               "expected": "query_trade_data"},
    {"question": "송파구 아파트 거래 알려줘",                   "expected": "query_trade_data"},
    {"question": "서초동 매매 최근 5건 알려줘",                 "expected": "query_trade_data"},

    # ── 전·월세 (행정구역) ───────────────────────────────────────────
    {"question": "역삼동 59㎡ 전세 시세 알려줘",               "expected": "query_rent_data"},
    {"question": "마포구 84㎡ 월세 알려줘",                    "expected": "query_rent_data"},
    {"question": "강남구 전세 최근 거래 보여줘",                "expected": "query_rent_data"},
    {"question": "합정동 월세 시세가 어떻게 돼?",               "expected": "query_rent_data"},
    {"question": "성북구 66㎡ 전월세 알려줘",                   "expected": "query_rent_data"},

    # ── 매매 (역·랜드마크) ───────────────────────────────────────────
    {"question": "강남역 근처 매매 거래 알려줘",                "expected": "query_trade_nearby"},
    {"question": "홍대입구역 주변 아파트 매매 시세",            "expected": "query_trade_nearby"},
    {"question": "코엑스 근처 아파트 거래 내역",                "expected": "query_trade_nearby"},
    {"question": "잠실역 주변 매매 최근 거래",                  "expected": "query_trade_nearby"},

    # ── 전·월세 (역·랜드마크) ────────────────────────────────────────
    {"question": "강남역 근처 전세 알려줘",                     "expected": "query_rent_nearby"},
    {"question": "홍대입구역 주변 월세 시세",                   "expected": "query_rent_nearby"},
    {"question": "판교역 근처 전월세 알려줘",                   "expected": "query_rent_nearby"},

    # ── 가격 예측 ────────────────────────────────────────────────────
    {"question": "마포구 84㎡ 아파트 가격 예측해줘",            "expected": "predict_price"},
    {"question": "강남구 59㎡ 10층 매매가 얼마나 될까?",        "expected": "predict_price"},
    {"question": "송파구 30평대 아파트 예상 매매가 알려줘",     "expected": "predict_price"},

    # ── 이상거래 탐지 ────────────────────────────────────────────────
    {"question": "강남구 이상거래 탐지해줘",                    "expected": "detect_anomaly"},
    {"question": "서초구에서 비정상 거래 있어?",                "expected": "detect_anomaly"},
    {"question": "마포구 이상 거래 분석해줘",                   "expected": "detect_anomaly"},

    # ── 지역 정보 ────────────────────────────────────────────────────
    {"question": "강남구 학군 정보 알려줘",                     "expected": "search_area_info"},
    {"question": "마포구 교통 편의시설 어때?",                  "expected": "search_area_info"},
    {"question": "서초구 살기 좋아?",                           "expected": "search_area_info"},
]


def run(verbose: bool = True) -> dict:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[3]))

    from src.agent.core.graph import build_graph

    graph = build_graph()
    results = []

    print(f"\n{'='*60}")
    print(f"도구 선택 정확도 평가  ({len(TEST_CASES)}개 질문)")
    print(f"{'='*60}")

    for i, tc in enumerate(TEST_CASES, 1):
        config = {"configurable": {"thread_id": f"eval_{i}"}}
        state = graph.invoke(
            {"messages": [HumanMessage(content=tc["question"])]},
            config=config,
        )
        # 호출된 도구 이름 추출
        called = []
        for msg in state["messages"]:
            for tc_call in getattr(msg, "tool_calls", []):
                if tc_call.get("name"):
                    called.append(tc_call["name"])

        first_tool = called[0] if called else "없음"
        correct = first_tool == tc["expected"]
        results.append(correct)

        if verbose:
            mark = "✓" if correct else "✗"
            print(f"  [{mark}] {tc['question'][:35]:<35} | 예상: {tc['expected']:<22} | 실제: {first_tool}")

    accuracy = sum(results) / len(results) * 100
    print(f"\n{'─'*60}")
    print(f"  정확도: {sum(results)}/{len(results)} = {accuracy:.1f}%")
    print(f"{'='*60}\n")
    return {"accuracy": accuracy, "correct": sum(results), "total": len(results)}


if __name__ == "__main__":
    run()
