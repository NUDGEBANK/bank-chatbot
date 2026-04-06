from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ChatRequest, ChatResponse
from .services import chat_service
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "NUDGEBANK AI 서버(FastAPI) 실행 중"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return StreamingResponse(
        chat_service.stream_answer(
            message=req.message,
            user_info=req.user_info
        ),
        media_type="text/plain; charset=utf-8"
    )