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

BANK_BACKEND_URL = os.getenv("BANK_BACKEND_URL", "http://localhost:9999")

class SuggestedAction(BaseModel):
    type: str
    label: str
    value: str | None = None
    href: str | None = None


class SuggestedActionBundle(BaseModel):
    quickReplies: list[SuggestedAction]


AVAILABLE_PAGES = [
    {"href": "/", "label": "홈으로 이동"},
    {"href": "/loan/products", "label": "대출 상품 보기"},
    {"href": "/loan/apply-guide", "label": "대출 신청 안내 보기"},
    {"href": "/deposit/products", "label": "예적금 상품 보기"},
    {"href": "/card/ddokgae", "label": "똑개 카드 보기"},
    {"href": "/card/spending-analysis", "label": "소비 분석 보기"},
    {"href": "/account/ddokgae", "label": "똑개 통장 보기"},
    {"href": "/help/chat-history", "label": "상담 기록 보기"},
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

        # Agent의 판단력을 높이기 위해 gpt-4o-mini 모델 유지
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500,
            streaming=True,
        )
        self.action_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=250,
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

    async def _infer_next_actions(
        self,
        user_message: str,
        bot_message: str,
    ) -> list[dict]:
        prompt = f"""
너는 NUDGEBANK 챗봇의 다음 행동 추천기다.
사용자 질문과 최종 답변을 보고, 다음에 누르면 좋은 버튼을 최대 3개 추천해라.

규칙:
- 버튼 타입은 ask 또는 navigate 둘 중 하나만 사용
- navigate는 반드시 아래 경로만 사용
- ask는 자연스러운 다음 질문이어야 함
- 현재 답변과 직접 이어지는 행동만 추천
- 중복 금지
- 한국어로 작성

사용 가능한 이동 경로:
{json.dumps(AVAILABLE_PAGES, ensure_ascii=False)}

사용자 질문:
{user_message}

최종 답변:
{bot_message}
""".strip()

        structured = self.action_llm.with_structured_output(SuggestedActionBundle)
        result = await structured.ainvoke(prompt)

        validated: list[dict] = []
        for item in result.quickReplies[:3]:
            if item.type == "navigate" and item.href:
                validated.append(
                    {
                        "type": "navigate",
                        "label": item.label,
                        "href": item.href,
                    }
                )
            elif item.type == "ask" and item.value:
                validated.append(
                    {
                        "type": "ask",
                        "label": item.label,
                        "value": item.value,
                    }
                )

        if validated:
            return validated

        return [
            {"type": "navigate", "label": "대출 상품 보기", "href": "/loan/products"},
            {"type": "navigate", "label": "상담 기록 보기", "href": "/help/chat-history"},
        ]

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
        loop = asyncio.get_event_loop()

        user_msg_embedding = await loop.run_in_executor(
            None, lambda: self.embed_model.encode(message).tolist()
        )

        @tool
        async def get_user_profile() -> str:
            """사용자 이름이나 현재 신용점수 등 개인 프로필 정보가 필요할 때 사용하세요."""
            return f"사용자 이름은 {name}이고, 현재 신용점수는 {credit}점입니다."

        @tool
        async def search_loan_info(query: str) -> str:
            """NUDGEBANK의 대출 상품, 금리, 조건, 신청 관련 정보에 대한 구체적인 지식이 필요할 때 이 도구를 사용하세요."""
            print(f"[Agent Tool Call] 🔍 대출 문서 검색 중: {query}") #LLM이 만든 쿼리와 비교
            query_embedding = await loop.run_in_executor(None, lambda: self.embed_model.encode(query).tolist())
            context = await loop.run_in_executor(None, self.vector_repository.search_documents, query_embedding)
            print(f"[문서 내용 (Doc RAG)]:\n{context}")
            return context

        @tool
        async def search_past_chat(query: str) -> str:
            """과거에 사용자와 나누었던 대화 기록이나 문맥을 확인해야 할 때 이 도구를 사용하세요."""
            # print(f"[Agent Tool Call] 🗂️ 과거 대화 검색 중: {query}")
            print(f"[Agent Tool Call] 🗂️ 과거 대화 검색 중: {message}") #유저 원문 메세지와 비교
            # query_embedding = await loop.run_in_executor(None, lambda: self.embed_model.encode(query).tolist())
            past_context = await loop.run_in_executor(None, self.chat_repository.search_past_conversations, member_id, session_id, user_msg_embedding)
            print(f"[📃대화 RAG (past_context)]:\n{past_context}")
            return past_context

        @tool
        async def check_loan_eligibility(product_key: str) -> str:
            """특정 상품의 대출 가능 여부를 실시간 조회한다. product_key는 youth-loan(자기계발 대출), consumption-loan(소비분석 대출), situate-loan(비상금 대출) 중 하나다."""
            print(f"[Agent Tool Call] 🔎 대출 가능 여부 조회 중: {product_key}")

            data = await fetch_loan_eligibility(
                access_token=access_token,
                product_key=product_key,
            )
            return build_eligibility_answer(data)

        tools = [
            get_user_profile,
            search_loan_info,
            search_past_chat,
            check_loan_eligibility,
        ]
        # 2. 시스템 프롬프트(페르소나) 정의
        system_prompt = """
당신은 NUDGEBANK 금융 상담 AI NUDGEBOT입니다.
답변은 도구를 활용하여 정확하지만 자연스럽게 답변하세요.
정확한 정보 제공을 위해 필요하다면 반드시 제공된 도구를 사용하세요.
특히 사용자 이름, 신용점수 같은 개인 정보가 필요하면 get_user_profile 도구를 사용하세요.
검색 도구를 사용한 후에도 관련 내용을 찾을 수 없다면, 임의로 지어내지 말고 정확한 확인을 위해 추가 참고가 필요하다고 안내하세요.
""".strip()

        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )

        try:


            # 기존 대화 기록 로드
            history = await loop.run_in_executor(None, self._build_history_messages, session_id)
            print(f"[📃대화 기록 (history)]:\n{history}")

            # 사용자 메시지 DB 저장
            await loop.run_in_executor(None, self._save_chat_message, session_id, "USER", message, user_msg_embedding)

            input_messages = history + [HumanMessage(content=message)]

            # 4. Agent 실행 및 스트리밍 처리
            async for event in agent.astream_events({"messages": input_messages}, version="v2"):
                # on_chat_model_stream 이벤트에서만 텍스트를 추출
                if event["event"] != "on_chat_model_stream":
                    continue
                chunk = event["data"]["chunk"]
                # 도구 호출(tool_calls) 데이터가 없고 순수 텍스트(content)만 있을 때 스트리밍
                if not chunk.content or chunk.tool_calls:
                    continue

                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        [c.get("text", "") for c in content if isinstance(c, dict)]
                    )

                if content:
                    bot_chunks.append(content)
                    yield to_sse("chunk", {"text": content})

            # 스트리밍 완료 후 봇 메시지 DB 저장
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

            quick_replies = await self._infer_next_actions(message, bot_message)
            yield to_sse(
                "done",
                {
                    "answer": bot_message,
                    "quickReplies": quick_replies,
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
