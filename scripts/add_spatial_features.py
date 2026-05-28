"""
apt_geocode 테이블에 dist_to_subway_km, dist_to_cityhall_km 컬럼 추가.

realestate/src/realestate/spatial.py의 좌표 데이터를 그대로 사용해
모든 단지에 대해 지하철역·구청까지 직선거리(Haversine)를 사전 계산합니다.

실행: python -m scripts.add_spatial_features
"""

import sys
from pathlib import Path

import numpy as np
from sqlalchemy import text

import json

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.agent.db.database import get_engine  # noqa: E402

_COORDS_PATH = Path(__file__).parents[1] / "src" / "agent" / "rag" / "spatial_coords.json"
with open(_COORDS_PATH, encoding="utf-8") as _f:
    _coords = json.load(_f)
SUBWAY_STATIONS  = {k: tuple(v) for k, v in _coords["subway_stations"].items()}
OFFICE_LOCATIONS = {k: tuple(v) for k, v in _coords["office_locations"].items()}


def _haversine_matrix(lats: np.ndarray, lngs: np.ndarray,
                      ref_lats: np.ndarray, ref_lngs: np.ndarray) -> np.ndarray:
    """(N_points, N_refs) 거리 행렬 반환 (km)"""
    R = 6371.0
    lats     = np.radians(lats)[:, None]
    lngs     = np.radians(lngs)[:, None]
    ref_lats = np.radians(ref_lats)[None, :]
    ref_lngs = np.radians(ref_lngs)[None, :]
    dlat = ref_lats - lats
    dlng = ref_lngs - lngs
    a = np.sin(dlat / 2) ** 2 + np.cos(lats) * np.cos(ref_lats) * np.sin(dlng / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main() -> None:
    engine = get_engine()

    # ── 컬럼 추가 (없으면) ────────────────────────────────────────────
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(apt_geocode)")).fetchall()}
        if "dist_to_subway_km" not in existing:
            conn.execute(text("ALTER TABLE apt_geocode ADD COLUMN dist_to_subway_km REAL"))
            print("컬럼 추가: dist_to_subway_km")
        if "dist_to_cityhall_km" not in existing:
            conn.execute(text("ALTER TABLE apt_geocode ADD COLUMN dist_to_cityhall_km REAL"))
            print("컬럼 추가: dist_to_cityhall_km")

    # ── 좌표 로드 ─────────────────────────────────────────────────────
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, latitude, longitude FROM apt_geocode WHERE latitude IS NOT NULL"
        )).fetchall()

    if not rows:
        print("좌표 데이터가 없습니다.")
        return

    ids  = np.array([r[0] for r in rows], dtype=int)
    lats = np.array([r[1] for r in rows], dtype=float)
    lngs = np.array([r[2] for r in rows], dtype=float)
    print(f"처리 대상: {len(ids):,}개 단지")

    # ── 거리 계산 ─────────────────────────────────────────────────────
    sta_arr = np.array(list(SUBWAY_STATIONS.values()))
    off_arr = np.array(list(OFFICE_LOCATIONS.values()))

    dist_subway = _haversine_matrix(lats, lngs, sta_arr[:, 0], sta_arr[:, 1]).min(axis=1).round(3)
    dist_office = _haversine_matrix(lats, lngs, off_arr[:, 0], off_arr[:, 1]).min(axis=1).round(3)
    print("거리 계산 완료")

    # ── DB 업데이트 ───────────────────────────────────────────────────
    params = [
        {"sub": float(d_sub), "off": float(d_off), "id": int(apt_id)}
        for apt_id, d_sub, d_off in zip(ids, dist_subway, dist_office)
    ]
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE apt_geocode SET dist_to_subway_km = :sub, dist_to_cityhall_km = :off WHERE id = :id"),
            params,
        )
    print(f"DB 업데이트 완료: {len(ids):,}개 단지")

    # ── 간단한 통계 출력 ──────────────────────────────────────────────
    print(f"\n  지하철 거리  평균 {dist_subway.mean():.2f}km / 중앙값 {np.median(dist_subway):.2f}km")
    print(f"  구청 거리    평균 {dist_office.mean():.2f}km / 중앙값 {np.median(dist_office):.2f}km")


if __name__ == "__main__":
    main()
