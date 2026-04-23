import asyncio
import httpx
import json
import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import HTTPException
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .repositories import ChatRepository, VectorRepository
from .schemas import LoanEligibilityResponse

load_dotenv()

BANK_BACKEND_URL = os.getenv("BANK_BACKEND_URL")

class SuggestedAction(BaseModel):
    type: str
    label: str
    value: str | None = None
    href: str | None = None

class SuggestedActionBundle(BaseModel):
    quickReplies: list[SuggestedAction]

AVAILABLE_PAGES = [
    {"href": "/", "label": "홈으로 이동"},

    {"href": "/about", "label": "은행 소개로 이동"},

    {"href": "/deposit/products", "label": "예적금 상품 페이지로 이동"},
    {"href": "/deposit/management", "label": "예적금 관리 페이지로 이동"},

    {"href": "/loan/products", "label": "대출 상품 페이지로 이동"},
    {"href": "/loan/products/consumption-loan", "label": "넛지 대출 상세 보기 페이지로 이동"},
    {"href": "/loan/products/consumption-loan/apply", "label": "넛지 대출 신청 페이지로 이동"},
    {"href": "/loan/products/youth-loan", "label": "자기계발 대출 상세 보기 페이지로 이동"},
    {"href": "/loan/products/youth-loan/apply", "label": "자기계발 대출 신청 페이지로 이동"},
    {"href": "/loan/apply-guide", "label": "대출 신청 안내 페이지로 이동"},
    {"href": "/loan/management", "label": "내 대출 관리 페이지로 이동"},
    {"href": "/loan/credit-score", "label": "신용 평가 페이지로 이동"},
    
    {"href": "/card/nudgecard", "label": "넛지 체크카드 페이지로 이동"},
    {"href": "/card/history", "label": "카드 이용 내역 페이지로 이동"},
    {"href": "/card/spending-analysis", "label": "소비 분석 페이지로 이동"},

    {"href": "/account/mypage", "label": "마이페이지로 이동"},
]

async def fetch_loan_eligibility(
    access_token: str,
    product_key: str,
) -> LoanEligibilityResponse:
    if not access_token:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{BANK_BACKEND_URL}/api/loan-products/eligibility",
            json={"productKey": product_key},
            headers={"Cookie": f"AT={access_token}"},
        )

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")

    if response.status_code >= 400:
        try:
            error_body = response.json()
        except Exception:
            error_body = {"message": response.text}

        raise HTTPException(
            status_code=response.status_code,
            detail=error_body.get("message", "대출 가능 여부 조회에 실패했습니다."),
        )

    return LoanEligibilityResponse(**response.json())

def build_eligibility_answer(data: LoanEligibilityResponse) -> str:
    if data.eligible:
        return (
            f"현재 회원 신용점수는 {data.creditScore}점입니다. "
            "대출 가능 기준을 충족하므로 신청 가능합니다."
        )

    reason = data.reasons[0] if data.reasons else "대출 기준을 충족하지 않습니다."
    return (
        f"현재 회원 신용점수는 {data.creditScore}점입니다. "
        f"{reason}"
    )

def to_sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

class ChatService:
    def __init__(self):
        self.chat_repository = ChatRepository()
        self.vector_repository = VectorRepository()
        self.embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

        # 이제 action_llm은 필요하지 않으므로 하나의 LLM만 유지합니다.
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1000,
            streaming=True,
        )

    def _load_session_messages(self, session_id: str) -> list[tuple[str, str]]:
        return self.chat_repository.load_session_messages(session_id)

    def _build_history_messages(self, session_id: str) -> list[HumanMessage | AIMessage]:
        history: list[HumanMessage | AIMessage] = []
        for sender_type, message_content in self._load_session_messages(session_id):
            if sender_type == "USER":
                history.append(HumanMessage(content=message_content))
            elif sender_type == "BOT":
                history.append(AIMessage(content=message_content))
        return history

    def prepare_chat_session(
        self, member_id: int, requested_session_id: str | None, first_message: str
    ) -> str:
        title = self._build_session_title(first_message)
        return self.chat_repository.prepare_chat_session(member_id, requested_session_id, title)

    def _build_session_title(self, message: str) -> str:
        compact = " ".join(message.split())
        if not compact:
            return "새 상담"
        return compact[:255]

    def _save_chat_message(
        self,
        session_id: str,
        sender_type: str,
        message_content: str,
        embedding: list[float] | None = None,
    ) -> None:
        self.chat_repository.save_chat_message(session_id, sender_type, message_content, embedding)

    async def stream_answer(
        self,
        message: str,
        user_info: dict,
        access_token: str,
    ) -> AsyncGenerator[str, None]:
        member_id = user_info.get("member_id")
        name = user_info.get("name", "고객")
        credit = user_info.get("creditScore", 0)
        session_id = user_info.get("session_id")

        if not session_id or not member_id:
            raise ValueError("session_id and member_id are required")

        bot_chunks: list[str] = []
        extracted_quick_replies: list[dict] = []
        loop = asyncio.get_event_loop()

        # 사용자 쿼리 임베딩
        user_msg_embedding = await loop.run_in_executor(
            None, lambda: self.embed_model.encode(message).tolist()
        )

        # 1. 도구(Tools) 정의
        @tool
        async def get_user_profile() -> str:
            """사용자 이름이나 현재 신용점수 등 개인 프로필 정보가 필요할 때 사용하세요."""
            return f"사용자 이름은 {name}이고, 현재 신용점수는 {credit}점입니다."

        @tool
        async def search_ragdocs_info(query: str) -> str:
            """NUDGEBANK의 은행 소개, 대출 상품, 예금 적금 상품, 카드, 금리, 조건, 신청 관련 정보, 기타 정보에 대한 구체적인 지식이 필요할 때 이 도구를 사용하세요."""
            print(f"[Agent Tool Call] 🔍 대출 문서 검색 중: {query}")
            query_embedding = await loop.run_in_executor(None, lambda: self.embed_model.encode(query).tolist())
            context = await loop.run_in_executor(None, self.vector_repository.search_documents, query_embedding)
            print(f"[문서 내용 (Doc RAG)]:\n{context}")
            return context

        @tool
        async def search_past_chat(query: str) -> str:
            """과거에 사용자와 나누었던 대화 기록이나 문맥을 확인해야 할 때 이 도구를 사용하세요."""
            print(f"[Agent Tool Call] 🗂️ 과거 대화 검색 중: {query}")
            query_embedding = await loop.run_in_executor(None, lambda: self.embed_model.encode(query).tolist())
            past_context = await loop.run_in_executor(None, self.chat_repository.search_past_conversations, member_id, session_id, query_embedding)
            print(f"[📃대화 RAG (past_context)]:\n{past_context}")
            return past_context

        @tool
        async def check_loan_eligibility(product_key: str) -> str:
            """특정 상품의 대출 가능 여부를 실시간 조회한다. product_key는 youth-loan(자기계발 대출), consumption-loan(넛지 대출) 중 하나다."""
            print(f"[Agent Tool Call] 🔎 대출 가능 여부 조회 중: {product_key}")
            data = await fetch_loan_eligibility(
                access_token=access_token,
                product_key=product_key,
            )
            return build_eligibility_answer(data)

        # [수정된 퀵 리플라이 도구]
        available_pages_info = "\n".join([f"- {p['label']}: {p['href']}" for p in AVAILABLE_PAGES])
        
        # suggest_quick_replies 설명
        quick_replies_description = f"""
        최종 텍스트 답변을 모두 작성한 뒤 마지막에 호출하세요.
        이 도구는 사용자에게 보여줄 quick reply 액션만 생성합니다.
        
        [필수 생성 규칙]
        1. 'ask' (후속 질문하기) 타입의 액션을 반드시 최소 2개 이상 포함하세요.
        2. 'navigate' (페이지 이동) 타입의 액션을 포함할 경우, 반드시 아래 [허용된 이동 경로] 목록에 있는 href 값만 사용해야 합니다.
        
        [허용된 이동 경로]
        {available_pages_info}
        """

        @tool(args_schema=SuggestedActionBundle, description=quick_replies_description, return_direct=True)
        async def suggest_quick_replies(quickReplies: list[SuggestedAction]) -> str:
            return "SUCCESS"

        tools = [
            get_user_profile,
            search_ragdocs_info,
            search_past_chat,
            check_loan_eligibility,
            # suggest_quick_replies
        ]

        # 2. 시스템 프롬프트(페르소나) 업데이트
        system_prompt = f"""
You are NUDGEBOT, a financial consulting AI agent for NUDGEBANK.
Your primary role is to assist users with banking services, loans, and personal financial information.

# Core Directives
1. Output Language: You MUST always respond in Korean (한국어).
2. Action Links (Absolute Trigger): No matter what the user says—whether it is a financial question, a simple greeting, a short reply, or an Out-of-Domain query—you MUST generate exactly 4 suggested action links at the very end of EVERY response, strictly following the Action Link Generation Rules below.
3. Tool Usage: You MUST use the `search_ragdocs_info` tool for EVERY question to gather domain knowledge. Use other tools as necessary.
4. Information Priority: Information retrieved from tools MUST take precedence over your pre-trained knowledge.
5. Missing Information: If you cannot find the relevant information even after using tools, DO NOT hallucinate or make up facts. Politely inform the user that additional checking is required.
6. Formatting: DO NOT output JSON, function call formats, or raw code blocks in your final response to the user.

# Strict Guardrails (Out-of-Domain Policy)
Your scope is strictly limited to NUDGEBANK's services, finance, user personal information, and chat history.
- You MUST politely decline any queries that are unrelated to these topics (Out-of-Domain).
- EXCEPTION: You are allowed to answer questions regarding the user's personal info, status, chat history, credit score, and loan eligibility (e.g., "Show my chat history", "What is my credit score?", "Can I get a loan?").

# Action Link Generation Rules (Strict Output Template)
At the very end of EVERY response, you MUST provide exactly 4 suggested actions. 
To ensure this, you MUST append the following exact Markdown structure at the bottom of your response without fail:

---
**💡 이런 질문은 어때요?**
- [First follow-up question](#ask=URL_ENCODED_QUESTION_1)
- [Second follow-up question](#ask=URL_ENCODED_QUESTION_2)

**🔗 바로가기**
- [Navigation Label 1](ALLOWED_PATH_1)
- [Navigation Label 2](ALLOWED_PATH_2)

* Rules for the Template:
1. The first two links MUST use the `#ask=` format. 
   CRITICAL: To ensure valid Markdown, DO NOT fully URL-encode the Korean text. You MUST keep the Korean characters as they are, but you MUST replace ALL spaces in the URL part with `%20`.
   (e.g., WRONG: [내 신용 점수](#ask=내 신용 점수) / CORRECT: [내 신용 점수](#ask=내%20신용%20점수))
2. The last two links MUST be selected ONLY from the [Allowed Paths for Navigation] list.
3. CRITICAL: DO NOT use `#` for the navigation links in the '바로가기' section. They MUST start exactly with `/` as provided in the list (e.g., `/loan/products`).
4. DO NOT change this visual layout. ALWAYS output exactly 2 questions and 2 navigation links.

[Allowed Paths for Navigation]
{available_pages_info}
""".strip()

        # 3. Agent 생성
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )

        try:
            history = await loop.run_in_executor(None, self._build_history_messages, session_id)
            print(f"[📃대화 기록 (history)]:\n{history}")

            await loop.run_in_executor(None, self._save_chat_message, session_id, "USER", message, user_msg_embedding)

            input_messages = history + [HumanMessage(content=message)]

            # 4. Agent 실행 및 스트리밍 (이벤트 가로채기)
            async for event in agent.astream_events({"messages": input_messages}, version="v2"):
                
                # 일반 텍스트 스트리밍
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if not chunk.content or chunk.tool_calls:
                        continue

                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join([c.get("text", "") for c in content if isinstance(c, dict)])

                    if content:
                        bot_chunks.append(content)
                        yield to_sse("chunk", {"text": content})

                # [NEW] 도구 호출을 가로채서 퀵 리플라이 데이터 추출
                elif event["event"] == "on_tool_start" and event["name"] == "suggest_quick_replies":
                    tool_input = event["data"].get("input", {})
                    # args_schema에 의해 배열로 전달된 데이터를 추출
                    qr_data = tool_input.get("quickReplies", [])
                    
                    for item in qr_data:
                        # Pydantic 모델 형태이거나 Dict 형태일 수 있으므로 안전하게 변환
                        qr_dict = item if isinstance(item, dict) else item.model_dump()
                        
                        # 유효성 검증
                        t = qr_dict.get("type")
                        if t == "navigate" and qr_dict.get("href"):
                            extracted_quick_replies.append(qr_dict)
                        elif t == "ask" and qr_dict.get("value"):
                            extracted_quick_replies.append(qr_dict)

            # 스트리밍 완료 후 봇 메시지 DB 저장 (텍스트만 깔끔하게 저장됨)
            bot_message = "".join(bot_chunks).strip()
            if bot_message:
                await loop.run_in_executor(
                    None,
                    self._save_chat_message,
                    session_id,
                    "BOT",
                    bot_message,
                    None,
                )

            # 만약 에이전트가 툴 호출을 누락했거나 형식이 잘못된 경우 기본값 세팅
            # if not extracted_quick_replies:
            #     extracted_quick_replies = [
            #         {"type": "navigate", "label": "대출 상품 보기", "href": "/loan/products"},
            #         {"type": "navigate", "label": "상담 기록 보기", "href": "/help/chat-history"},
            #     ]

            # 최종적으로 'done' 이벤트에 텍스트와 퀵 리플라이를 함께 반환
            yield to_sse(
                "done",
                {
                    "answer": bot_message,
                    "quickReplies": extracted_quick_replies,
                },
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Error in stream_answer: {exc}")
            yield to_sse(
                "error",
                {"message": "죄송합니다. 응답 생성 중 오류가 발생했습니다."},
            )

    def _get_user_profile(self, member_id: int):
        return self.chat_repository.get_user_profile(member_id)

    def list_chat_sessions(self, member_id: int) -> list[dict]:
        return self.chat_repository.list_chat_sessions(member_id)

    def get_chat_session_detail(self, member_id: int, session_id: str) -> dict:
        return self.chat_repository.get_chat_session_detail(member_id, session_id)

    def rename_chat_session(self, member_id: int, session_id: str, title: str) -> dict:
        return self.chat_repository.rename_chat_session(member_id, session_id, title)

    def delete_chat_session(self, member_id: int, session_id: str) -> None:
        self.chat_repository.delete_chat_session(member_id, session_id)

chat_service = ChatService()
