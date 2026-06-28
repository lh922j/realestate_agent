import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import httpx
import pandas as pd
import pydeck as pdk
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from src.agent.core.graph import build_graph
from src.agent.config import settings
from src.agent.tools.query import fetch_map_points
from src.agent.tools.nearby import fetch_map_points_nearby
from src.agent.tools.query_rent import fetch_map_points_rent

# ─── 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="부동산 AI 어시스턴트",
    page_icon="🏠",
    layout="wide",
)

# ─── 그래프 초기화 ────────────────────────────────────────────────
@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()

# ─── 세션 초기화 ─────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "map_entries" not in st.session_state:
    st.session_state.map_entries = []

# ─── 상수 ────────────────────────────────────────────────────────
TOOL_LABELS = {
    "query_trade_data":    "📊 매매 실거래 데이터 조회 중...",
    "query_rent_data":     "📊 전·월세 데이터 조회 중...",
    "query_trade_nearby":  "📍 주변 매매 실거래 조회 중...",
    "query_rent_nearby":   "📍 주변 전·월세 실거래 조회 중...",
    "predict_price":       "🤖 가격 예측 모델 실행 중...",
    "detect_anomaly":      "🔍 이상거래 분석 중...",
    "search_area_info":    "📚 지역 정보 검색 중...",
}

# ─── 헬퍼: 지도 좌표 추출 ─────────────────────────────────────────
def _extract_map_points(all_messages: list) -> list[dict]:
    from src.agent.tools.nearby import _geocode
    for msg in reversed(all_messages):
        for tc in getattr(msg, "tool_calls", []):
            a = tc["args"]
            if tc["name"] == "query_trade_data":
                pts = fetch_map_points(
                    district=a.get("district", ""),
                    area_min=a.get("area_min", 0),
                    area_max=a.get("area_max", 300),
                    year_from=a.get("year_from", 2024),
                    year_to=a.get("year_to", 2026),
                )
                if pts:
                    return pts
            elif tc["name"] == "query_trade_nearby":
                lat = a.get("latitude", 0)
                lon = a.get("longitude", 0)
                if not lat and a.get("place_name"):
                    coords = _geocode(a["place_name"])
                    if coords:
                        lat, lon = coords
                if lat and lon:
                    pts = fetch_map_points_nearby(
                        latitude=lat, longitude=lon,
                        radius_km=a.get("radius_km", 1.0),
                        area_min=a.get("area_min", 0),
                        area_max=a.get("area_max", 300),
                        year_from=a.get("year_from", 2024),
                        year_to=a.get("year_to", 2026),
                    )
                    if pts:
                        return pts
            elif tc["name"] == "query_rent_data":
                pts = fetch_map_points_rent(
                    district=a.get("district", ""),
                    area_min=a.get("area_min", 0),
                    area_max=a.get("area_max", 300),
                    rent_type=a.get("rent_type", "전체"),
                    year_from=a.get("year_from", 2024),
                    year_to=a.get("year_to", 2026),
                )
                if pts:
                    return pts
            elif tc["name"] == "query_rent_nearby":
                lat = a.get("latitude", 0)
                lon = a.get("longitude", 0)
                if not lat and a.get("place_name"):
                    coords = _geocode(a["place_name"])
                    if coords:
                        lat, lon = coords
                if lat and lon:
                    from src.agent.tools.nearby import fetch_map_points_nearby as _fmn
                    pts = _fmn(
                        latitude=lat, longitude=lon,
                        radius_km=a.get("radius_km", 1.0),
                        area_min=a.get("area_min", 0),
                        area_max=a.get("area_max", 300),
                        year_from=a.get("year_from", 2024),
                        year_to=a.get("year_to", 2026),
                    )
                    if pts:
                        return pts
    return []


# ─── 헬퍼: 지하철역 조회 (Kakao 카테고리 SW8) ───────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_nearby_stations(lat: float, lon: float, radius_m: int = 1500) -> list[dict]:
    if not settings.kakao_api_key:
        return []
    try:
        resp = httpx.get(
            "https://dapi.kakao.com/v2/local/search/category.json",
            params={"category_group_code": "SW8", "x": lon, "y": lat,
                    "radius": radius_m, "size": 15},
            headers={"Authorization": f"KakaoAK {settings.kakao_api_key}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        return [
            {"name": d["place_name"], "lat": float(d["y"]), "lon": float(d["x"])}
            for d in resp.json().get("documents", [])
        ]
    except Exception:
        return []



# ─── 지도 렌더링 ─────────────────────────────────────────────────
def _render_map(entry: dict):
    df = pd.DataFrame(entry["points"])
    # 전세·월세는 price_label 필드가 이미 있고, 매매는 새로 생성
    if "price_label" not in df.columns:
        df["price_label"] = df["deal_amount"].apply(lambda x: f"{int(x):,}만원")

    # 가격 기반 색상 (낮음=파랑, 높음=빨강)
    p_min, p_max = df["deal_amount"].min(), df["deal_amount"].max()
    p_range = max(p_max - p_min, 1)
    df["r"] = ((df["deal_amount"] - p_min) / p_range * 200 + 55).astype(int)
    df["g"] = 55
    df["b"] = (255 - (df["deal_amount"] - p_min) / p_range * 200).astype(int)

    center_lat = round(df["latitude"].mean(), 4)
    center_lon = round(df["longitude"].mean(), 4)

    layers: list = []

    # ── 1. 아파트 레이어 ─────────────────────────────────────────
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_fill_color=["r", "g", "b", 210],
        get_radius=80,
        radius_min_pixels=6,
        radius_max_pixels=20,
        pickable=True,
    ))

    # ── 3. 지하철역 레이어 (핀 스타일) ───────────────────────────
    stations = _fetch_nearby_stations(center_lat, center_lon)
    if stations:
        sdf = pd.DataFrame(stations)

        # 외곽 원 (주황 테두리 + 흰 채움)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=sdf,
            get_position=["lon", "lat"],
            get_fill_color=[255, 255, 255, 240],
            get_line_color=[255, 100, 0, 255],
            stroked=True,
            filled=True,
            line_width_min_pixels=3,
            get_radius=90,
            radius_min_pixels=11,
            radius_max_pixels=22,
            pickable=True,
        ))
        # 내부 점 (주황)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=sdf,
            get_position=["lon", "lat"],
            get_fill_color=[255, 100, 0, 255],
            get_radius=35,
            radius_min_pixels=4,
            radius_max_pixels=8,
            pickable=False,
        ))
        # 역 이름 라벨 (흰 배경)
        layers.append(pdk.Layer(
            "TextLayer",
            data=sdf,
            get_position=["lon", "lat"],
            get_text="name",
            get_size=11,
            get_color=[180, 60, 0, 255],
            get_pixel_offset=[0, -18],
            background=True,
            get_background_color=[255, 255, 255, 210],
            get_border_width=1,
            get_border_color=[255, 180, 100, 200],
            get_padding=[4, 2, 4, 2],
            pickable=False,
        ))

    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=14, pitch=0)

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={
                "html": (
                    "<b>{apt_name}{name}</b><br/>"
                    "{dong_name}<br/>"
                    "{area_exclusive}㎡&nbsp;&nbsp;💰 {price_label}"
                ),
                "style": {
                    "backgroundColor": "#1e1e2e",
                    "color": "white",
                    "fontSize": "13px",
                    "padding": "6px 10px",
                },
            },
        ),
        use_container_width=True,
    )

    disp = df[["apt_name", "dong_name", "area_exclusive", "deal_amount"]].copy()
    disp.columns = ["아파트명", "동명", "면적(㎡)", "거래금액(만원)"]
    disp["거래금액(만원)"] = disp["거래금액(만원)"].apply(lambda x: f"{int(x):,}")
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ─── UI ──────────────────────────────────────────────────────────
st.title("🏠 부동산 AI 어시스턴트")
st.caption("한국 아파트 실거래가 분석 · 가격 예측 · 이상거래 탐지")

with st.sidebar:
    st.header("ℹ️ 사용 안내")
    st.markdown("""
- **매매** 조회: "역삼동 59㎡ 매매 거래 알려줘"
- **전세/월세** 조회: "역삼동 59㎡ 전세 알려줘"
- 매매·전세·월세 조회 시 **오른쪽 지도** 자동 표시
""")
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.map_entries = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.divider()
    debug_mode = st.toggle("🐛 디버그 모드", value=False)

    # ── 질문 기록 ───────────────────────────────────────────────
    st.subheader("📝 질문 기록")
    past_questions = [
        msg.content
        for msg in st.session_state.messages
        if isinstance(msg, HumanMessage)
    ]
    if past_questions:
        for i, q in enumerate(reversed(past_questions)):
            label = q[:22] + "…" if len(q) > 22 else q
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.pending_input = q
    else:
        st.caption("아직 질문 기록이 없습니다")

    st.divider()
    st.caption("Powered by LangGraph + GPT-4o-mini")

st.markdown("**빠른 질문 예시**")
btn_cols = st.columns(4)
examples = [
    "역삼동 59㎡ 매매 최근 거래 알려줘",
    "역삼동 59㎡ 전세 시세 알려줘",
    "마포구 84㎡ 아파트 가격 예측해줘",
    "강남구 이상거래 탐지해줘",
]
for col, ex in zip(btn_cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state.pending_input = ex

st.divider()

# ─── 채팅 입력 ───────────────────────────────────────────────────
pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("질문을 입력하세요...") or pending

# ─── 2컬럼 레이아웃 ──────────────────────────────────────────────
chat_col, map_col = st.columns([3, 2])

with chat_col:
    chat_container = st.container(height=650, border=False)

    with chat_container:
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage) and msg.content:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            input_len = len(st.session_state.messages)
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                status_slot = st.empty()
                status_slot.info("🔍 분석 중...")
                text_slot   = st.empty()
                full_text   = ""
                debug_steps = []

                _stream_modes = ["messages", "values", "updates"] if debug_mode else ["messages", "values"]

                try:
                    for event in graph.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        stream_mode=_stream_modes,
                        config=config,
                    ):
                        mode, data = event

                        if mode == "values":
                            final_state = data

                        elif mode == "updates" and debug_mode:
                            debug_steps.append(data)

                        elif mode == "messages":
                            chunk, meta = data
                            node = meta.get("langgraph_node", "")

                            if node == "agent" and isinstance(chunk, AIMessageChunk):
                                valid_tools = [
                                    tc["name"] for tc in getattr(chunk, "tool_calls", [])
                                    if tc.get("name")
                                ]
                                if valid_tools:
                                    labels = [TOOL_LABELS.get(n, f"🔄 {n}") for n in valid_tools]
                                    status_slot.info(" · ".join(labels))
                                elif chunk.content:
                                    status_slot.empty()
                                    full_text += chunk.content
                                    text_slot.markdown(full_text + "▌")

                except Exception as e:
                    status_slot.empty()
                    text_slot.error(f"오류가 발생했습니다: {e}")
                    st.stop()

                status_slot.empty()
                text_slot.markdown(full_text)

                if debug_mode and debug_steps:
                    with st.expander("🐛 실행 추적", expanded=True):
                        for step in debug_steps:
                            for node, changes in step.items():
                                st.markdown(f"**`[{node}]`**")
                                for m in changes.get("messages", []):
                                    if hasattr(m, "tool_calls") and m.tool_calls:
                                        for tc in m.tool_calls:
                                            st.code(f"→ {tc['name']}\n{tc['args']}", language="python")
                                    elif hasattr(m, "content") and m.content:
                                        preview = str(m.content)
                                        st.code(preview[:400] + ("..." if len(preview) > 400 else ""))

            if final_state:
                all_msgs = final_state["messages"]
                new_msgs = all_msgs[input_len:]  # 현재 턴에서 새로 생긴 메시지만
                st.session_state.messages.extend(new_msgs)

                # 현재 턴 메시지만 전달 — 이전 턴 역삼동 등 다른 지역 tool call 재사용 방지
                pts = _extract_map_points(new_msgs)
                if pts:
                    st.session_state.map_entries.append({
                        "question": user_input,
                        "points":   pts,
                    })

with map_col:
    if not st.session_state.map_entries:
        st.info("🗺️ 매매 거래를 조회하면 지도가 자동으로 표시됩니다")
    else:
        for i, entry in enumerate(reversed(st.session_state.map_entries)):
            label = entry["question"][:30] + ("..." if len(entry["question"]) > 30 else "")
            with st.expander(f"📍 {label}", expanded=(i == 0)):
                _render_map(entry)
