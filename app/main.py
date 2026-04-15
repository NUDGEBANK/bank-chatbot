import base64
import os

import jwt
from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from typing import Optional

from .schemas import ChatRequest, ChatResponse, ChatSessionDetail, ChatSessionSummary
from .services import (
    chat_service,
)

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
        "https://nudgebank.shinhanacademy.co.kr"
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


@app.post("/chat-api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    member_id = extract_member_id_from_cookie(request)
    access_token = request.cookies.get("AT")

    req.user_info["member_id"] = member_id

    try:
        profile = chat_service._get_user_profile(member_id)
        req.user_info["name"] = profile["name"]
        req.user_info["creditScore"] = profile["credit"]
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        print(f"_get_user_profile error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to load user profile") from exc

    try:
        session_id = chat_service.prepare_chat_session(
            member_id=member_id,
            requested_session_id=req.session_id,
            first_message=req.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid session_id") from exc
    except Exception as exc:
        print(f"prepare_chat_session error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to prepare chat session") from exc

    req.user_info["session_id"] = session_id

    return StreamingResponse(
        chat_service.stream_answer(
            message=req.message,
            user_info=req.user_info,
            access_token=access_token,
        ),
        media_type="text/plain; charset=utf-8",
        headers={"X-Chat-Session-Id": session_id},
    )


@app.get("/chat-api/chat/sessions", response_model=list[ChatSessionSummary])
def get_chat_sessions(request: Request):
    member_id = extract_member_id_from_cookie(request)

    try:
        return chat_service.list_chat_sessions(member_id)
    except Exception as exc:
        print(f"get_chat_sessions error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to load chat sessions") from exc


@app.get("/chat-api/chat/sessions/{session_id}", response_model=ChatSessionDetail)
def get_chat_session(session_id: str, request: Request):
    member_id = extract_member_id_from_cookie(request)

    try:
        return chat_service.get_chat_session_detail(member_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found") from exc
    except Exception as exc:
        print(f"get_chat_session error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to load chat session") from exc


@app.patch("/chat-api/chat/sessions/{session_id}", response_model=ChatSessionSummary)
async def rename_chat_session(session_id: str, request: Request):
    member_id = extract_member_id_from_cookie(request)

    try:
        payload = await request.json()
        title = str(payload.get("title", ""))
        return chat_service.rename_chat_session(member_id, session_id, title)
    except ValueError as exc:
        message = str(exc)
        if message == "chat session not found":
            raise HTTPException(status_code=404, detail="Chat session not found") from exc
        raise HTTPException(status_code=400, detail="Invalid chat session title") from exc
    except Exception as exc:
        print(f"rename_chat_session error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to rename chat session") from exc


@app.delete("/chat-api/chat/sessions/{session_id}", status_code=204)
def delete_chat_session(session_id: str, request: Request):
    member_id = extract_member_id_from_cookie(request)

    try:
        chat_service.delete_chat_session(member_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Chat session not found") from exc
    except Exception as exc:
        print(f"delete_chat_session error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to delete chat session") from exc
