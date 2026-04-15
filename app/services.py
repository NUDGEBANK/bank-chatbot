import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from .repositories import ChatRepository, VectorRepository

load_dotenv()


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

    def _save_chat_message(self, session_id: str, sender_type: str, message_content: str, embedding: list[float] | None = None) -> None:
        self.chat_repository.save_chat_message(session_id, sender_type, message_content, embedding)

    async def stream_answer(self, message: str, user_info: dict) -> AsyncGenerator[str, None]:
        member_id = user_info.get("member_id")
        name = user_info.get("name", "고객")
        credit = user_info.get("creditScore", 0)
        session_id = user_info.get("session_id")

        if not session_id or not member_id:
            raise ValueError("session_id and member_id are required")

        bot_chunks: list[str] = []
        loop = asyncio.get_event_loop()

        # 1. 도구(Tools) 정의: session_id와 member_id를 사용하기 위해 함수 내부에 선언
        @tool
        async def search_loan_info(query: str) -> str:
            """NUDGEBANK의 대출 상품, 금리, 조건 등에 대한 구체적인 지식이 필요할 때 이 도구를 사용하세요."""
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

        # 에이전트가 사용할 도구 목록
        tools = [search_loan_info, search_past_chat]

        # 2. 시스템 프롬프트(페르소나) 정의
        system_prompt = f"""당신은 NUDGEBANK 금융 상담 AI NUDGEBOT입니다.
사용자의 이름은 {name}이며, 신용점수는 {credit}입니다.
답변은 핵심만 간단명료하게 작성하세요.
정확한 정보 제공을 위해 필요하다면 반드시 제공된 검색 도구(대출 정보 검색, 과거 대화 검색)를 활용하세요.
검색 도구를 사용한 후에도 관련된 내용을 찾을 수 없다면, 임의로 지어내지 말고 정확한 확인을 위해 추가 참고가 필요하다고 안내하세요."""

        # 3. Agent 생성
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )

        try:
            # 사용자 쿼리 임베딩
            user_msg_embedding = await loop.run_in_executor(None, lambda: self.embed_model.encode(message).tolist())
            
            # 기존 대화 기록 로드
            history = await loop.run_in_executor(None, self._build_history_messages, session_id)
            print(f"[📃대화 기록 (history)]:\n{history}")

            # 사용자 메시지 DB 저장
            await loop.run_in_executor(None, self._save_chat_message, session_id, "USER", message, user_msg_embedding)
            
            input_messages = history + [HumanMessage(content=message)]

            # 4. Agent 실행 및 스트리밍 처리
            async for event in agent.astream_events({"messages": input_messages}, version="v2"):
                # on_chat_model_stream 이벤트에서만 텍스트를 추출
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    
                    # 도구 호출(tool_calls) 데이터가 없고 순수 텍스트(content)만 있을 때 스트리밍
                    if chunk.content and not chunk.tool_calls:
                        content = chunk.content
                        if isinstance(content, list):
                            content = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
                        
                        bot_chunks.append(content)
                        yield content

            # 스트리밍 완료 후 봇 메시지 DB 저장
            bot_message = "".join(bot_chunks).strip()
            if bot_message:
                await loop.run_in_executor(None, self._save_chat_message, session_id, "BOT", bot_message, None)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Error in stream_answer: {exc}")
            yield "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."

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