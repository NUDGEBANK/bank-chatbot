import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    message: str
    user_info: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"message": "FastAPI AI 서버 실행 중"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    message = req.message
    user_info = req.user_info

    income = user_info.get("income", 0)
    credit = user_info.get("creditScore", 0)

    # 🔥 챗봇 로직 (초기 버전)
    if "대출" in message:
        if income > 4000 and credit > 700:
            answer = "대출 승인 가능성이 높습니다."
        else:
            answer = "대출 심사가 필요합니다."
    else:
        answer = f"'{message}'에 대한 답변입니다."

    return ChatResponse(answer=answer)