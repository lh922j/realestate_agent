# 부동산 Agentic AI

한국 아파트 실거래가 분석 · 가격 예측 · 이상거래 탐지를 위한 Agentic AI 서비스.

> **연계 프로젝트**: [realestate](https://github.com/lh922j/realestate) — 국토교통부 API로 수집한 실거래 데이터 ETL 파이프라인 및 LightGBM 예측 모델 학습.  
> 이 프로젝트는 realestate에서 구축한 SQLite DB와 학습된 모델 파일(`.pkl`)을 직접 활용합니다.

---

## 서비스 특징

**채팅 + 지도를 동시에 제공하는 부동산 AI 어시스턴트**입니다.

일반적인 챗봇과 달리, 거래 조회 결과를 텍스트 답변과 함께 **우측 지도 패널에 자동으로 시각화**합니다.

```
┌─────────────────────────┬──────────────────────────┐
│  AI 채팅 영역            │  지도 영역               │
│                         │                          │
│  Q: 역삼동 59㎡ 매매    │  • 거래 아파트 위치 점   │
│     거래 알려줘          │    (가격 낮음=파랑,      │
│                         │     높음=빨강)            │
│  A: [표 형태 답변]      │  • 주변 지하철역 오버레이 │
│                         │  • 툴팁: 아파트명/가격    │
│  Q: 강남역 근처 전세는? │                          │
│  A: ...                 │  [이전 질문 지도 접기]   │
└─────────────────────────┴──────────────────────────┘
```

- 매매 · 전세 · 월세 조회 시 지도 자동 표시 (행정구역 이름 또는 역·랜드마크 기반 모두 지원)
- 여러 턴에 걸친 조회 기록을 **Expander로 누적 표시** — 이전 결과와 비교 가능
- Kakao API 연동 시 **주변 지하철역**을 지도에 함께 표시

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

## 평가 지표

정답 레이블 없이 측정 가능한 세 가지 지표로 평가합니다.  
(`python -m tests.eval.run_all` 으로 전체 재현 가능)

### 1. 도구 선택 정확도

26개 질문에 대해 에이전트가 올바른 도구를 선택하는지 측정합니다.

| | 결과 |
|--|------|
| 정확도 | **96.2%** (25 / 26) |
| 실패 케이스 | "서초구 살기 좋아?" — 의도가 모호해 도구 미호출 |

### 2. 가격 예측 정확도

전체 단지를 **단지 단위(80/20)**로 분리하여 평가합니다.  
테스트셋(20%)은 학습에 전혀 포함되지 않은 단지로 구성 — 일반화 성능 측정.

| 지표 | 값 |
|------|----|
| 테스트 단지 수 | 3,254개 단지 (196,035건) |
| MAE  | 16,507만원 (1.65억원) |
| RMSE | 28,740만원 (2.87억원) |
| R²   | 0.714 |
| MAPE | 23.6% |

> 수도권 아파트 가격 분포가 5천만~100억 이상으로 극단적으로 넓어  
> MAE 절댓값이 크게 보이지만, 중저가 단지 기준으로는 오차가 더 작습니다.

### 3. 도구 레이턴시 (3회 평균)

| 도구 | 평균 응답 시간 |
|------|--------------|
| `query_trade_data` | 0.02s |
| `query_trade_nearby` | 0.18s |
| `predict_price` | 0.75s |
| `detect_anomaly` | 0.67s |
| `query_rent_data` | 2.87s |
| `search_area_info` | 0.02s (모델 웜업 후) |

> `search_area_info`는 BGE-M3 첫 로드 시 ~13초 소요, 이후 캐시되어 0.02s.  
> `query_rent_data`는 전월세 데이터 규모(300만건+)로 인해 상대적으로 느림.

---

## 기술 스택

- **LangGraph** — ReAct 에이전트 그래프, 멀티턴 메모리
- **GPT-4o-mini** — LLM 추론
- **LightGBM** — 아파트 가격 예측 (realestate/ 프로젝트에서 학습)
- **Isolation Forest** — 이상거래 탐지
- **ChromaDB + BGE-M3** — 지역 정보 벡터 검색 (로컬 임베딩)
- **SQLite / SQLAlchemy** — 실거래 데이터 저장 및 조회
- **Streamlit + pydeck** — 대화형 대시보드 및 지도 시각화
