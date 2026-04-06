from fastapi import FastAPI
from .schemas import ChatRequest, ChatResponse
from .services import chat_service

app = FastAPI()

@app.get("/")
def root():
    return {"message": "NUDGEBANK AI 서버(FastAPI) 실행 중"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 서비스 레이어에 로직 위임
    answer = await chat_service.get_answer(
        message=req.message, 
        user_info=req.user_info
    )
    
    return ChatResponse(answer=answer)