from typing import Any, Dict, Optional

from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    message: str
    user_info: Dict[str, Any]
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
