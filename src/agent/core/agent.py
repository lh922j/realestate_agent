from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState
from ..config import settings
from ..tools import get_tools

SYSTEM_PROMPT = """당신은 한국 부동산 실거래가 분석 전문 AI 어시스턴트입니다.

## 답변 스타일
- 핵심만 간결하게 3~5줄로 답변하세요. 불필요한 설명은 생략합니다.
- 거래 목록은 상위 5건만 표로 요약하세요.
- 이상하거나 주의할 점만 한 줄로 추가하세요.

## 도구 선택 규칙 (절대 준수)
동·구 이름이 포함된 경우 (예: 역삼동, 강남구, 작전동 — "근처"가 붙어도 동일):
- 매매 조회 → query_trade_data (district에 동/구 이름 입력)
- 전세·월세 조회 → query_rent_data (district에 동/구 이름 입력, rent_type: '전세'/'월세'/'전체')

역·건물·랜드마크 등 행정구역이 아닌 장소 이름:
- 매매 조회 → query_trade_nearby (place_name에 장소명 직접 입력)
- 전세·월세 조회 → query_rent_nearby (place_name에 장소명 직접 입력)
- 예: place_name="작전역", place_name="코엑스" — 좌표는 자동으로 조회됩니다

기타:
- 가격 예측 → predict_price
- 지역 정보(학군·교통·편의시설) → search_area_info
- 이상거래 탐지 → detect_anomaly

## 중요 원칙 (절대 준수)
- 매매와 전세·월세는 완전히 다른 데이터입니다. 절대 혼용하지 마세요.
- 새 질문이 들어오면 반드시 도구를 새로 호출하세요. 이전 턴의 조회 결과로 답하는 것은 금지입니다.
- 질문의 지역·조건이 이전 턴과 달라졌다면 무조건 도구를 다시 호출하세요.
- 도구를 호출하지 않고는 가격을 절대 추측하거나 답변하지 마세요.
- 부동산과 무관한 질문에는 "저희 Agent와 무관한 질문입니다, 아파트와 관련된 질문 부탁드려요"라고 답변하세요.

## 검색 규칙
- 가격은 만원 단위로 표기합니다
- "84㎡" 조회 시 area_min=80, area_max=90 범위로 검색하세요
- 최근 거래 조회 시 특별한 언급 없으면 year_from=2024, year_to=2026 사용

"""

_SYSTEM_MSG = SystemMessage(content=SYSTEM_PROMPT)


@lru_cache(maxsize=1)
def _get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=800,
        api_key=settings.openai_api_key,
    ).bind_tools(get_tools())


def _keep_recent(messages: list, max_turns: int = 6) -> list:
    """HumanMessage 기준으로 최근 N턴만 유지. 첫 번째 HumanMessage는 항상 포함."""
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_indices) <= max_turns:
        return messages
    cutoff = human_indices[-max_turns]
    return messages[cutoff:]


def agent_node(state: AgentState) -> dict:
    messages = [_SYSTEM_MSG] + _keep_recent(state.messages)
    response = _get_llm().invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state.messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"
