"""
태스크 완료율 평가 (end-to-end extrinsic)

에이전트 최종 응답이 질문 의도에 맞는 답을 포함하는지 측정합니다.
tool_selection.py의 도구 선택(intrinsic)과 달리, 실제 답변 내용을 검증합니다.
실행: python -m tests.eval.task_completion
"""

import re

_price   = lambda r: bool(re.search(r"\d[\d,]*\s*(만원|억원|억)", r))
_anomaly = lambda r: any(kw in r for kw in ["이상", "비정상", "탐지"]) and bool(re.search(r"\d+", r))

TEST_CASES = [
    # ── 매매 (행정구역) ──────────────────────────────────────────────
    {"question": "역삼동 59㎡ 매매 최근 거래 알려줘",          "label": "매매 조회",     "check": _price},
    {"question": "강남구 84㎡ 아파트 매매 내역 보여줘",         "label": "매매 조회",     "check": _price},
    {"question": "마포구 59㎡ 매매 시세 알려줘",               "label": "매매 조회",     "check": _price},
    {"question": "송파구 아파트 거래 알려줘",                   "label": "매매 조회",     "check": _price},
    {"question": "서초동 매매 최근 5건 알려줘",                 "label": "매매 조회",     "check": _price},

    # ── 전·월세 (행정구역) ───────────────────────────────────────────
    {"question": "역삼동 59㎡ 전세 시세 알려줘",               "label": "전월세 조회",   "check": _price},
    {"question": "마포구 84㎡ 월세 알려줘",                    "label": "전월세 조회",   "check": _price},
    {"question": "강남구 전세 최근 거래 보여줘",                "label": "전월세 조회",   "check": _price},
    {"question": "합정동 월세 시세가 어떻게 돼?",               "label": "전월세 조회",   "check": _price},
    {"question": "성북구 66㎡ 전월세 알려줘",                   "label": "전월세 조회",   "check": _price},

    # ── 매매 (역·랜드마크) ───────────────────────────────────────────
    {"question": "강남역 근처 매매 거래 알려줘",                "label": "주변 매매",     "check": _price},
    {"question": "홍대입구역 주변 아파트 매매 시세",            "label": "주변 매매",     "check": _price},
    {"question": "코엑스 근처 아파트 거래 내역",                "label": "주변 매매",     "check": _price},
    {"question": "잠실역 주변 매매 최근 거래",                  "label": "주변 매매",     "check": _price},

    # ── 전·월세 (역·랜드마크) ────────────────────────────────────────
    {"question": "강남역 근처 전세 알려줘",                     "label": "주변 전월세",   "check": _price},
    {"question": "홍대입구역 주변 월세 시세",                   "label": "주변 전월세",   "check": _price},
    {"question": "판교역 근처 전월세 알려줘",                   "label": "주변 전월세",   "check": _price},

    # ── 가격 예측 ────────────────────────────────────────────────────
    {"question": "마포구 84㎡ 아파트 가격 예측해줘",            "label": "가격 예측",     "check": lambda r: ("예측" in r or "예상" in r) and _price(r)},
    {"question": "강남구 59㎡ 10층 매매가 얼마나 될까?",        "label": "가격 예측",     "check": _price},
    {"question": "송파구 30평대 아파트 예상 매매가 알려줘",     "label": "가격 예측",     "check": _price},

    # ── 이상거래 탐지 ────────────────────────────────────────────────
    {"question": "강남구 이상거래 탐지해줘",                    "label": "이상거래 탐지", "check": _anomaly},
    {"question": "서초구에서 비정상 거래 있어?",                "label": "이상거래 탐지", "check": _anomaly},
    {"question": "마포구 이상 거래 분석해줘",                   "label": "이상거래 탐지", "check": _anomaly},

    # ── 지역 정보 ────────────────────────────────────────────────────
    {"question": "강남구 학군 정보 알려줘",                     "label": "지역 정보",     "check": lambda r: "강남" in r and any(kw in r for kw in ["학교", "학군", "교육", "학원"])},
    {"question": "마포구 교통 편의시설 어때?",                  "label": "지역 정보",     "check": lambda r: "마포" in r and any(kw in r for kw in ["교통", "지하철", "버스", "편의"])},
    {"question": "서초구 살기 좋아?",                           "label": "지역 정보",     "check": lambda r: any(kw in r for kw in ["서초", "교통", "편의", "학군", "생활"])},
]


def run(verbose: bool = True) -> dict:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[3]))

    from langchain_core.messages import AIMessage, HumanMessage

    from src.agent.core.graph import build_graph

    graph = build_graph()
    results = []

    print(f"\n{'='*60}")
    print(f"태스크 완료율 평가  ({len(TEST_CASES)}개 질문)")
    print(f"{'='*60}")

    for i, tc in enumerate(TEST_CASES, 1):
        config = {"configurable": {"thread_id": f"tc_eval_{i}"}}
        state = graph.invoke(
            {"messages": [HumanMessage(content=tc["question"])]},
            config=config,
        )

        final_response = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        passed = tc["check"](final_response)
        results.append({"passed": passed, "label": tc["label"]})

        if verbose:
            mark = "✓" if passed else "✗"
            preview = final_response[:40].replace("\n", " ")
            print(f"  [{mark}] [{tc['label']:<8}] {tc['question'][:30]:<30} | {preview}")

    # ── 전체 및 유형별 집계 ──────────────────────────────────────────
    total = len(results)
    correct = sum(r["passed"] for r in results)
    accuracy = correct / total * 100

    labels = sorted({r["label"] for r in results})
    print(f"\n{'─'*60}")
    for label in labels:
        sub = [r for r in results if r["label"] == label]
        sub_correct = sum(r["passed"] for r in sub)
        print(f"  {label:<12}: {sub_correct}/{len(sub)}")
    print(f"{'─'*60}")
    print(f"  전체 완료율: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*60}\n")

    return {"accuracy": accuracy, "correct": correct, "total": total}


if __name__ == "__main__":
    run()
