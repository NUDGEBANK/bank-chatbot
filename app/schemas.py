from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
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


class RagDocumentSummary(BaseModel):
    loan_product_id: int
    source_name: str
    chunk_count: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RagIngestResponse(BaseModel):
    status: str
    message: str
    assigned_product_id: int
    document: Optional[RagDocumentSummary] = None
    logs: list[str]


class RagDeleteResponse(BaseModel):
    message: str
    deleted_chunks: int
    logs: list[str]
class LoanEligibilityResponse(BaseModel):
    eligible: bool
    decision: str
    creditScore: int
    productKey: str
    reasons: list[str]
