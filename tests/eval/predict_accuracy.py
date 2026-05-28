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
    # 학습과 동일: apt_key = apt_name + '_' + lawd_cd 기준 LEFT JOIN
    sql = text("""
        SELECT t.apt_name, t.dong_name, t.lawd_cd,
               t.area_exclusive, t.floor, t.build_year, t.building_age, t.deal_amount,
               t.deal_year, t.deal_month, t.sgg_code, t.dealing_type,
               g.latitude, g.longitude,
               g.dist_to_subway_km, g.dist_to_cityhall_km
        FROM apt_trade t
        LEFT JOIN apt_geocode g ON (t.apt_name || '_' || t.lawd_cd) = g.apt_key
        WHERE t.deal_amount > 0
        ORDER BY t.deal_date
    """)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    df = pd.DataFrame(rows, columns=[
        "apt_name", "dong_name", "lawd_cd",
        "area_exclusive", "floor", "build_year", "building_age", "deal_amount",
        "deal_year", "deal_month", "sgg_code", "dealing_type",
        "latitude", "longitude",
        "dist_to_subway_km", "dist_to_cityhall_km",
    ])
    # deal_amount·build_year 등 핵심 수치만 dropna (좌표 NaN은 중앙값으로 후처리)
    df = df.dropna(subset=["deal_amount", "build_year", "area_exclusive", "floor"])

    print(f"  전체: {len(df):,}건")

    # ── 단지 단위 분리 (학습 시 apt_key = apt_name + lawd_cd 와 동일하게) ────
    df["apt_key"] = df["apt_name"] + "_" + df["lawd_cd"].astype(str)
    all_keys = df["apt_key"].unique()

    rng = np.random.default_rng(random_seed)
    all_keys = np.array(all_keys)   # ArrowStringArray → numpy (shuffle 보장)
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

    # ── sgg_code 변환: lawd_cd(API) → GeoJSON 코드 (학습 시 공간 조인과 동일 형식) ─
    # 학습 시 add_sgg_info()가 GeoJSON 경계 공간 조인으로 sgg_code를 OVERWRITE함.
    # GeoJSON 코드(예: 강남구="11230")와 API lawd_cd(예: "11680")가 다른 형식이므로
    # 사전 계산된 매핑으로 변환.  좌표가 없는 행은 None → "unknown" (학습과 동일).
    import json as _json
    _mapping_path = Path(__file__).parents[2] / "src" / "agent" / "rag" / "lawd_cd_to_sgg_code.json"
    with open(_mapping_path, encoding="utf-8") as _f:
        _lawd_to_geo = _json.load(_f)

    # 좌표가 있는 행만 GeoJSON sgg_code 적용 (없으면 None → 인코더에서 "unknown")
    has_coord = df["latitude"].notna()
    df["sgg_code"] = None
    df.loc[has_coord, "sgg_code"] = (
        df.loc[has_coord, "lawd_cd"].astype(str).map(_lawd_to_geo)
    )

    # ── 피처 생성 (전체 df 기준으로 먼저 처리 → 학습 시 중앙값 일치) ────────
    gangnam_lat, gangnam_lon = 37.4979, 127.0276
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}

    def haversine_series(d):
        R = 6371.0
        dlat = np.radians(gangnam_lat - d["latitude"])
        dlon = np.radians(gangnam_lon - d["longitude"])
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(d["latitude"])) * np.cos(np.radians(gangnam_lat)) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    # building_age: DB 저장값 사용 (학습 시 2025-build_year 기준, deal_year 무관)
    df["deal_season"]        = df["deal_month"].map(season_map)
    df["dist_to_gangnam_km"] = haversine_series(df)

    # NaN 채움: 전체 데이터 중앙값 사용 (학습 시 build_feature_matrix와 동일)
    for col in ["latitude", "longitude", "dist_to_gangnam_km", "dist_to_subway_km", "dist_to_cityhall_km"]:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    for col, enc_map in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(
                lambda x, m=enc_map: m.get(x, m.get("unknown", 0))
            )

    # 분할 후 테스트셋 추출
    test_df = df[df["apt_key"].isin(test_keys)].copy()

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
