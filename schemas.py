# schemas.py
from pydantic import BaseModel
from typing import Any, Dict


class UserRequest(BaseModel):
    user_text: str
    session_id: str


class FinalResponse(BaseModel):
    input_type: str
    transcribed_text: str | None = None
    original_text: str | None = None
    response: Dict[str, Any]


class ChatHistoryItem(BaseModel):
    chatroom_id: str
    content: str


class AnalysisResponse(BaseModel):
    """분석 결과 응답 모델"""

    persona: str
    reasoning: str
    feedback: str
