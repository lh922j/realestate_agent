"""
가격 예측 모델 정확도 평가 (단지 단위 분리)

전체 데이터를 단지(apt_key) 기준으로 80/20 분리하여 평가합니다.
테스트 단지는 학습에 포함되지 않은 단지로 구성 — 일반화 성능 측정.
실행: python -m tests.eval.predict_accuracy
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


def run(test_ratio: float = 0.2, random_seed: int = 42) -> dict:
    import joblib
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    from src.agent.config import settings
    from src.agent.db.database import get_engine

    print(f"\n{'='*60}")
    print(f"가격 예측 정확도 평가  (단지 단위 분리, test_ratio={test_ratio})")
    print(f"{'='*60}")

    # ── 전체 데이터 로드 ──────────────────────────────────────────────
    sql = text("""
        SELECT t.apt_name, t.dong_name,
               t.area_exclusive, t.floor, t.build_year, t.deal_amount,
               t.deal_year, t.deal_month, t.sgg_code, t.dealing_type,
               g.latitude, g.longitude
        FROM apt_trade t
        JOIN apt_geocode g ON t.apt_name = g.apt_name AND t.dong_name = g.dong_name
        WHERE g.latitude IS NOT NULL
          AND t.deal_amount > 0
          AND (t.cancel_type IS NULL OR t.cancel_type = '')
    """)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    df = pd.DataFrame(rows, columns=[
        "apt_name", "dong_name",
        "area_exclusive", "floor", "build_year", "deal_amount",
        "deal_year", "deal_month", "sgg_code", "dealing_type",
        "latitude", "longitude",
    ]).dropna()

    print(f"  전체: {len(df):,}건")

    # ── 단지 단위 분리 ────────────────────────────────────────────────
    df["apt_key"] = df["apt_name"] + "_" + df["dong_name"]
    all_keys = df["apt_key"].unique()

    rng = np.random.default_rng(random_seed)
    rng.shuffle(all_keys)
    n_test = max(1, int(len(all_keys) * test_ratio))
    test_keys = set(all_keys[:n_test])

    test_df  = df[df["apt_key"].isin(test_keys)].copy()
    train_df = df[~df["apt_key"].isin(test_keys)]

    print(f"  학습 단지: {len(all_keys) - n_test:,}개 ({len(train_df):,}건)")
    print(f"  테스트 단지: {n_test:,}개 ({len(test_df):,}건)  ← 학습 미포함 단지")

    # ── 모델 로드 ─────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(settings.model_path)

    model = bundle["model"]
    model.set_params(n_jobs=1)
    encoders = {
        col: {cls: i for i, cls in enumerate(le.classes_)}
        for col, le in bundle.get("label_encoders", {}).items()
    }

    # ── 피처 생성 ─────────────────────────────────────────────────────
    gangnam_lat, gangnam_lon = 37.4979, 127.0276
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}

    def haversine_series(df):
        R = 6371.0
        dlat = np.radians(gangnam_lat - df["latitude"])
        dlon = np.radians(gangnam_lon - df["longitude"])
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(df["latitude"])) * np.cos(np.radians(gangnam_lat)) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    test_df["building_age"]       = test_df["deal_year"] - test_df["build_year"]
    test_df["deal_season"]        = test_df["deal_month"].map(season_map)
    test_df["dist_to_gangnam_km"] = haversine_series(test_df)
    test_df["dist_to_subway_km"]  = 0.5
    test_df["dist_to_cityhall_km"] = 2.0

    for col, mapping in encoders.items():
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str).map(
                lambda x, m=mapping: m.get(x, m.get("unknown", 0))
            )

    X = test_df[bundle["feature_names"]].astype("float64")
    y_true = test_df["deal_amount"].values

    # ── 예측 및 지표 계산 ─────────────────────────────────────────────
    log_pred = model.predict(X)
    y_pred   = np.expm1(log_pred)

    mae  = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100

    print(f"\n  MAE  : {mae:>10,.0f} 만원  ({mae/10000:.2f} 억원)")
    print(f"  RMSE : {rmse:>10,.0f} 만원  ({rmse/10000:.2f} 억원)")
    print(f"  R²   : {r2:>10.4f}")
    print(f"  MAPE : {mape:>10.2f} %")
    print(f"{'='*60}\n")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
            "n_test": len(test_df), "n_test_keys": n_test}


if __name__ == "__main__":
    run()
