# 부동산 Agentic AI

한국 아파트 실거래가 분석 · 가격 예측 · 이상거래 탐지를 위한 Agentic AI 서비스.

> **연계 프로젝트**: [realestate](https://github.com/lh922j/realestate) — 국토교통부 API로 수집한 실거래 데이터 ETL 파이프라인 및 LightGBM 예측 모델 학습.  
> 이 프로젝트는 realestate에서 구축한 SQLite DB와 학습된 모델 파일(`.pkl`)을 직접 활용합니다.

---

## 프로젝트 구조

```
realestate/          ← 연계 프로젝트 (데이터 수집 · 모델 학습)
│  data/processed/realestate.db     ← SQLite DB (매매 · 전월세)
│  data/models/price_model_trade_lgbm_complex.pkl  ← 예측 모델
│
realestate_agent/    ← 이 프로젝트 (Agentic AI 서비스)
├── app/
│   └── streamlit_app.py    # Streamlit 대시보드
├── src/agent/
│   ├── core/
│   │   ├── agent.py        # LLM 노드 (GPT-4o-mini + System Prompt)
│   │   ├── graph.py        # LangGraph ReAct 그래프
│   │   └── state.py        # 대화 상태
│   ├── tools/
│   │   ├── query.py        # 매매 실거래 조회 (SQLite)
│   │   ├── query_rent.py   # 전·월세 조회 (SQLite)
│   │   ├── nearby.py       # 위치 기반 주변 거래 조회
│   │   ├── predict.py      # LightGBM 가격 예측
│   │   ├── anomaly.py      # Isolation Forest 이상거래 탐지
│   │   └── rag.py          # 지역 정보 벡터 검색 (ChromaDB)
│   ├── rag/
│   │   ├── embeddings.py   # BGE-M3 임베딩 (로컬)
│   │   ├── indexer.py      # ChromaDB 인덱싱
│   │   └── area_info.json  # 지역별 교통·학군·개발 정보
│   ├── db/
│   │   └── database.py     # SQLAlchemy 엔진
│   └── config.py           # 환경변수 설정 (pydantic-settings)
└── src/agent/api/
    └── main.py             # FastAPI 서버 (선택)
```

---

## 아키텍처

```
사용자 질문
    │
    ▼
[LLM Agent] ── GPT-4o-mini + System Prompt (LangGraph ReAct)
    │
    │ 도구 선택
    ▼
┌────────────────────────────────────────────────┐
│  SQL 조회          ML / 분석        RAG         │
│  query_trade_data  predict_price   search_area  │
│  query_rent_data   detect_anomaly  _info        │
│  query_*_nearby                                 │
└────────────────────────────────────────────────┘
    │
    ▼
SQLite DB          LightGBM pkl      ChromaDB
(realestate/)      (subprocess)      (BGE-M3, 로컬)
    │
    ▼
[LLM] 최종 답변 + Streamlit 지도 시각화
```

- **다중 턴 대화**: LangGraph `MemorySaver` checkpointer로 세션 유지
- **가격 예측**: segfault 방지를 위해 `subprocess.run`으로 LightGBM 격리 실행
- **지도**: pydeck `ScatterplotLayer` + Kakao API 지하철역 오버레이

---

## 사전 요구사항

1. **[realestate](https://github.com/lh922j/realestate) 프로젝트**를 먼저 클론하고 데이터 수집 및 모델 학습을 완료해야 합니다.

   ```bash
   # realestate/ 프로젝트에서 실행
   python -m src.realestate.main collect --type trade --region 서울 --start 202201 --end 202604
   python -m src.realestate.main collect --type rent  --region 서울 --start 202201 --end 202604
   python -m src.realestate.main geocode
   python -m src.realestate.main train --model lgbm --complex-split
   ```

2. Python 3.11 이상

---

## 설치 및 실행

```bash
git clone https://github.com/lh922j/realestate_agent.git
cd realestate_agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`.env` 파일 생성 (`.env.example` 참고):

```bash
cp .env.example .env
# .env 편집: OPENAI_API_KEY, DATABASE_URL, MODEL_PATH 설정
```

```bash
# Streamlit 실행
streamlit run app/streamlit_app.py

# FastAPI 서버 실행 (선택)
uvicorn src.agent.api.main:app --reload
```

---

## 주요 기능

| 기능 | 사용 도구 | 예시 질문 |
|------|-----------|-----------|
| 매매 실거래 조회 | `query_trade_data` | "역삼동 59㎡ 매매 거래 알려줘" |
| 전·월세 실거래 조회 | `query_rent_data` | "마포구 84㎡ 전세 시세 알려줘" |
| 위치 기반 주변 조회 | `query_trade_nearby` / `query_rent_nearby` | "강남역 근처 월세 알려줘" |
| 아파트 가격 예측 | `predict_price` | "마포구 84㎡ 아파트 가격 예측해줘" |
| 이상거래 탐지 | `detect_anomaly` | "강남구 이상거래 탐지해줘" |
| 지역 정보 검색 | `search_area_info` | "서초구 학군 정보 알려줘" |

---

## 환경변수

| 변수 | 설명 | 필수 |
|------|------|------|
| `OPENAI_API_KEY` | GPT-4o-mini API 키 | ✅ |
| `DATABASE_URL` | SQLite/PostgreSQL DB 경로 | ✅ |
| `MODEL_PATH` | LightGBM `.pkl` 파일 경로 | ✅ |
| `KAKAO_API_KEY` | 지오코딩 · 지하철 조회 | 선택 |
| `CHROMA_PATH` | ChromaDB 저장 경로 (기본 `./data/chroma`) | 선택 |

---

## 기술 스택

- **LangGraph** — ReAct 에이전트 그래프, 멀티턴 메모리
- **GPT-4o-mini** — LLM 추론
- **LightGBM** — 아파트 가격 예측 (realestate/ 프로젝트에서 학습)
- **Isolation Forest** — 이상거래 탐지
- **ChromaDB + BGE-M3** — 지역 정보 벡터 검색 (로컬 임베딩)
- **SQLite / SQLAlchemy** — 실거래 데이터 저장 및 조회
- **Streamlit + pydeck** — 대화형 대시보드 및 지도 시각화
