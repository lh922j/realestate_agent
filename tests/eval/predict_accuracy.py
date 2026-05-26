"""
가격 예측 모델 정확도 평가

DB의 실거래 데이터를 정답으로 삼아 LightGBM 모델 성능을 측정합니다.
실행: python -m tests.eval.predict_accuracy
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))


def run(year_from: int = 2025, year_to: int = 2026, sample_n: int = 500) -> dict:
    import joblib
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    from src.agent.config import settings
    from src.agent.db.database import get_engine

    print(f"\n{'='*60}")
    print(f"가격 예측 정확도 평가  ({year_from}~{year_to}년 데이터, 최대 {sample_n}건)")
    print(f"{'='*60}")

    # ── 테스트 데이터 로드 ────────────────────────────────────────────
    sql = text("""
        SELECT t.area_exclusive, t.floor, t.build_year, t.deal_amount,
               t.deal_year, t.deal_month, t.sgg_code, t.dealing_type,
               g.latitude, g.longitude
        FROM apt_trade t
        JOIN apt_geocode g ON t.apt_name = g.apt_name AND t.dong_name = g.dong_name
        WHERE t.deal_year BETWEEN :year_from AND :year_to
          AND g.latitude IS NOT NULL
          AND t.deal_amount > 0
          AND (t.cancel_type IS NULL OR t.cancel_type = '')
        ORDER BY RANDOM()
        LIMIT :n
    """)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year_from": year_from, "year_to": year_to, "n": sample_n}).fetchall()

    if not rows:
        print("  평가 데이터 없음 — DB에 해당 기간 데이터가 없습니다.")
        return {}

    df = pd.DataFrame(rows, columns=[
        "area_exclusive", "floor", "build_year", "deal_amount",
        "deal_year", "deal_month", "sgg_code", "dealing_type",
        "latitude", "longitude",
    ])
    df = df.dropna()
    print(f"  로드: {len(df)}건")

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

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    df["building_age"] = df["deal_year"] - df["build_year"]
    df["deal_season"] = df["deal_month"].map(season_map)
    df["dist_to_gangnam_km"] = df.apply(
        lambda r: haversine(r.latitude, r.longitude, gangnam_lat, gangnam_lon), axis=1
    )
    df["dist_to_subway_km"] = 0.5
    df["dist_to_cityhall_km"] = 2.0

    for col, mapping in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: mapping.get(x, mapping.get("unknown", 0)))

    X = df[bundle["feature_names"]].astype("float64")
    y_true = df["deal_amount"].values

    # ── 예측 및 지표 계산 ─────────────────────────────────────────────
    log_pred = model.predict(X)
    y_pred = np.expm1(log_pred)

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

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "n": len(df)}


if __name__ == "__main__":
    run()
