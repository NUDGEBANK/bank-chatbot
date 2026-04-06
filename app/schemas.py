from pydantic import BaseModel
from typing import Dict, Any

class ChatRequest(BaseModel):
    user_id: str
    message: str
    user_info: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str