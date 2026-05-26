from langchain_core.tools import tool
from loguru import logger
from sqlalchemy import text

from ..db.database import get_engine


@tool
def detect_anomaly(
    district: str,
    area_min: float = 0,
    area_max: float = 300,
    year_from: int = 2022,
    year_to: int = 2025,
    contamination: float = 0.02,
) -> str:
    """
    특정 지역·면적 조건의 거래에서 Isolation Forest로 이상거래를 탐지합니다.

    Args:
        district: 동명 또는 구명 (예: '역삼동', '강남구')
        area_min: 전용면적 최솟값 (㎡)
        area_max: 전용면적 최댓값 (㎡)
        year_from: 조회 시작 연도
        year_to: 조회 종료 연도
        contamination: 이상치 예상 비율 (0.01~0.1, 기본 0.02)
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    logger.info(f"[anomaly] district={district} area={area_min}~{area_max}㎡ year={year_from}~{year_to} contamination={contamination}")

    sql = text("""
        SELECT t.apt_name, t.dong_name, t.area_exclusive, t.floor,
               t.deal_amount, t.price_per_sqm, t.building_age,
               t.deal_year, t.deal_month, t.sgg_code, t.deal_date,
               g.latitude, g.longitude
        FROM apt_trade t
        LEFT JOIN apt_geocode g
               ON t.apt_name = g.apt_name AND t.dong_name = g.dong_name
        WHERE t.dong_name LIKE :district
          AND t.area_exclusive BETWEEN :area_min AND :area_max
          AND t.deal_year BETWEEN :year_from AND :year_to
          AND (t.cancel_type IS NULL OR t.cancel_type = '')
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
            }).fetchall()

        if not rows:
            return f"'{district}' 지역에서 분석할 데이터가 없습니다."

        cols = ["apt_name", "dong_name", "area_exclusive", "floor",
                "deal_amount", "price_per_sqm", "building_age",
                "deal_year", "deal_month", "sgg_code", "deal_date",
                "latitude", "longitude"]
        df = pd.DataFrame(rows, columns=cols)

        for c in ["deal_amount", "price_per_sqm", "area_exclusive", "floor",
                  "building_age", "deal_year", "deal_month", "latitude", "longitude"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        logger.debug(f"[anomaly] 조회 {len(df):,}건")

        # Isolation Forest 피처 (사용 가능한 것만)
        feature_candidates = [
            "deal_amount", "price_per_sqm", "area_exclusive", "floor",
            "building_age", "deal_year", "deal_month",
            "latitude", "longitude",
        ]
        features = [c for c in feature_candidates if c in df.columns and df[c].notna().sum() > 0]

        X = df[features].copy()
        for col in features:
            X[col] = X[col].fillna(X[col].median())

        X_scaled = StandardScaler().fit_transform(X)

        clf = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled)
        scores = clf.score_samples(X_scaled)
        preds = clf.predict(X_scaled)

        df["anomaly_score"] = scores.round(4)
        df["is_anomaly"] = preds == -1

        n_anomaly = df["is_anomaly"].sum()
        logger.info(f"[anomaly] 이상거래 {n_anomaly}건 / {len(df):,}건 ({n_anomaly/len(df)*100:.1f}%)")

        summary = (
            f"분석 대상: {len(df):,}건 | "
            f"이상거래 탐지 (Isolation Forest): {n_anomaly}건 ({n_anomaly/len(df)*100:.1f}%)\n"
        )

        if n_anomaly == 0:
            return summary + "이상거래가 발견되지 않았습니다."

        # 이상 점수 낮은 순(더 이상한 순)으로 상위 10건
        top = df[df["is_anomaly"]].sort_values("anomaly_score").head(10)

        lines = [summary]
        for _, r in top.iterrows():
            lines.append(
                f"{r['apt_name']} {r['dong_name']} {r['area_exclusive']:.1f}㎡ "
                f"{int(r['floor'])}층 → {int(r['deal_amount']):,}만원 "
                f"(㎡당 {int(r['price_per_sqm']):,}만, score={r['anomaly_score']:.3f}, {r['deal_date']})"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[anomaly] 오류: {e}")
        return f"이상탐지 오류: {e}"
