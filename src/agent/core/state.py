from typing import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class MapPoint(BaseModel):
    apt_name: str
    dong_name: str
    area_exclusive: float
    deal_amount: float
    latitude: float
    longitude: float


class AgentState(BaseModel):
    """LangGraph 에이전트 상태"""
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    map_points: list[MapPoint] = Field(default_factory=list)
