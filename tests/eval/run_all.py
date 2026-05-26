"""
전체 평가 실행 스크립트

실행: python -m tests.eval.run_all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


def main():
    from tests.eval import predict_accuracy, latency

    print("\n" + "=" * 60)
    print("  부동산 Agentic AI — 전체 평가")
    print("=" * 60)

    # 1. 가격 예측 정확도
    pred = predict_accuracy.run()

    # 2. 레이턴시 (tool_selection은 LLM 호출 비용이 발생하므로 선택적 실행)
    lat = latency.run()

    # ── 요약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  최종 요약")
    print("=" * 60)

    if pred:
        print(f"  [가격 예측]  MAE {pred['mae']:,.0f}만원 | R² {pred['r2']:.4f} | MAPE {pred['mape']:.1f}%")

    if lat:
        slowest = max(lat, key=lambda k: lat[k]["avg"])
        fastest = min(lat, key=lambda k: lat[k]["avg"])
        print(f"  [레이턴시]   최빠름: {fastest}({lat[fastest]['avg']:.2f}s) | 최느림: {slowest}({lat[slowest]['avg']:.2f}s)")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
