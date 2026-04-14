import base64
import os

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from typing import Optional
from fastapi import Cookie, FastAPI, HTTPException

from .schemas import ChatRequest, ChatResponse, ChatSessionDetail, ChatSessionSummary, ChatMessageRequest
from .services import (
    build_eligibility_answer,
    chat_service,
    infer_intent,
    infer_product_key,
    fetch_loan_eligibility,
    build_eligibility_api_context,
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

    api_context = ""

    try:
        intent = infer_intent(req.message)

        if intent == "loan_eligibility_check":
            product_key = infer_product_key(req.message)
            if product_key:
                token = request.cookies.get("AT")
                eligibility = await fetch_loan_eligibility(
                    access_token=token,
                    product_key=product_key,
                )
                api_context = build_eligibility_api_context(eligibility)
            else:
                api_context = (
                    "대출 가능 여부 조회 질문이지만 상품 정보가 없습니다. "
                    "자기계발 대출, 소비분석 대출, 비상금 대출 중 어떤 상품인지 먼저 확인이 필요합니다."
                )
    except Exception as exc:
        print(f"loan eligibility api context error: {exc}")
        api_context = "대출 가능 여부 조회 API 결과를 가져오지 못했습니다."

    return StreamingResponse(
        chat_service.stream_answer(
            message=req.message,
            user_info=req.user_info,
            api_context=api_context,
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

@app.post("/chat/message")
async def chat_message(
    body: ChatMessageRequest,
    AT: Optional[str] = Cookie(default=None),
):
    if not AT:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다.")

    intent = infer_intent(message)

    if intent != "loan_eligibility_check":
        return {
            "answer": "현재는 대출 가능 여부 조회 질문만 처리할 수 있습니다.",
            "intent": intent,
        }

    product_key = body.productKey or infer_product_key(message)
    if not product_key:
        return {
            "answer": "어떤 상품 기준으로 확인할까요? 자기계발 대출, 소비분석 대출, 비상금 대출 중에서 말씀해 주세요.",
            "intent": intent,
        }

    eligibility = await fetch_loan_eligibility(
        access_token=AT,
        product_key=product_key,
    )
    answer = build_eligibility_answer(eligibility)

    return {
        "answer": answer,
        "intent": intent,
        "eligibility": eligibility.model_dump(),
    }
