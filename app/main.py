import base64
import os

import jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from .schemas import ChatRequest, ChatResponse
from .services import chat_service

load_dotenv()

app = FastAPI()

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

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
    return {"message": "NUDGEBANK AI server is running"}


def extract_member_id_from_cookie(request: Request) -> int:
    token = request.cookies.get("AT")
    if not token:
        print("Chat auth failed: missing AT cookie")
        raise HTTPException(status_code=401, detail="No token")

    if not JWT_SECRET:
        print("Chat auth failed: JWT_SECRET is not configured")
        raise HTTPException(status_code=500, detail="JWT secret is not configured")

    try:
        secret_key = base64.b64decode(JWT_SECRET)
        payload = jwt.decode(token, secret_key, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except (ValueError, KeyError, jwt.PyJWTError) as exc:
        print(f"Chat auth failed: invalid token ({exc})")
        raise HTTPException(status_code=401, detail="Invalid token") from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    member_id = extract_member_id_from_cookie(request)
    req.user_info["session_id"] = str(member_id)
    try:
        profile = chat_service._get_user_profile(member_id)
        req.user_info["name"] = profile["name"]
        req.user_info["creditScore"] = profile["credit"]
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        print(f"_get_user_profile error: {exc}")
        raise HTTPException(status_code=500, detail="사용자 조회 중 오류가 발생했습니다.") from exc

    return StreamingResponse(
        chat_service.stream_answer(
            message=req.message,
            user_info=req.user_info,
        ),
        media_type="text/plain; charset=utf-8",
    )
