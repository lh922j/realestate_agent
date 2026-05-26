from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel

from ..core.graph import build_graph


# ─── 앱 시작 시 그래프 컴파일 ────────────────────────────────────
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("LangGraph 컴파일 중...")
    _graph = build_graph()
    logger.info("LangGraph 준비 완료")
    yield
    logger.info("서버 종료")


app = FastAPI(
    title="부동산 AI 에이전트",
    description="한국 아파트 실거래가 분석 LangGraph 에이전트 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 스키마 ──────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

    model_config = {"json_schema_extra": {"example": {"message": "강남구 84㎡ 아파트 최근 시세 알려줘"}}}


class ChatResponse(BaseModel):
    answer: str


# ─── 엔드포인트 ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"[API] /chat 요청: {request.message[:80]}")
    result = _graph.invoke({"messages": [HumanMessage(content=request.message)]})
    answer = result["messages"][-1].content
    logger.info(f"[API] /chat 응답: {answer[:80]}")
    return ChatResponse(answer=answer)
