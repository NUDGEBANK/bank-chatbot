from datetime import datetime
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    message: str
    user_info: Dict[str, Any]
    session_id: Optional[str] = None


class ChatActionAsk(BaseModel):
    type: Literal["ask"]
    label: str
    value: str


class ChatActionNavigate(BaseModel):
    type: Literal["navigate"]
    label: str
    href: str

ChatAction = Union[ChatActionAsk, ChatActionNavigate]


class ChatResponse(BaseModel):
    answer: str
    quickReplies: list[ChatAction] = []


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ChatMessageItem(BaseModel):
    message_id: int
    sender_type: str
    message_content: str
    created_at: Optional[datetime] = None


class ChatSessionDetail(BaseModel):
    session_id: str
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    messages: list[ChatMessageItem]

class LoanEligibilityResponse(BaseModel):
    eligible: bool
    decision: str
    creditScore: int
    productKey: str
    reasons: list[str]
