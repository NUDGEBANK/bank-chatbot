import asyncio
import os
import uuid
from typing import AsyncGenerator

import psycopg2
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()


class ChatService:
    def __init__(self):
        self.embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
당신은 NUDGEBANK 금융 상담 AI입니다.
사용자의 이름은 {name}이며 현재 정보는 다음과 같습니다:
- 신용점수: {credit}
답변은 핵심만 간단명료하게 작성하세요.
아래 제공된 [문서 내용]을 바탕으로 사용자의 질문에 친절하고 구체적으로 답변하세요.
만약 문서와 관련된 내용이 없다면 그 범위 안에서는 정확한 확인을 위해 추가 참고가 필요하다고 안내하세요.

[문서 내용]
{context}
""".strip(),
                ),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{message}"),
            ]
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500,
            streaming=True,
        )
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

    def _get_db_connection(self):
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
        )
        register_vector(conn)
        return conn

    def _load_session_messages(self, session_id: str) -> list[tuple[str, str]]:
        conn = self._get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT sender_type, message_content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, message_id ASC
                """,
                (session_id,),
            )
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()

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
        session_id = self._resolve_session_id(requested_session_id)
        title = self._build_session_title(first_message)

        conn = self._get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT title
                FROM chat_sessions
                WHERE session_id = %s AND member_id = %s
                """,
                (session_id, member_id),
            )
            existing = cur.fetchone()

            if existing is None:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (session_id, member_id, title, created_at, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (session_id, member_id, title),
                )
            else:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET
                        title = COALESCE(title, %s),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s AND member_id = %s
                    """,
                    (title, session_id, member_id),
                )

            conn.commit()
            return session_id
        finally:
            cur.close()
            conn.close()

    def _resolve_session_id(self, requested_session_id: str | None) -> str:
        if not requested_session_id:
            return str(uuid.uuid4())
        return str(uuid.UUID(requested_session_id))

    def _build_session_title(self, message: str) -> str:
        compact = " ".join(message.split())
        if not compact:
            return "새 상담"
        return compact[:255]

    def _save_chat_message(self, session_id: str, sender_type: str, message_content: str) -> None:
        conn = self._get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO chat_messages (session_id, sender_type, message_content, created_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (session_id, sender_type, message_content),
            )
            cur.execute(
                """
                UPDATE chat_sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = %s
                """,
                (session_id,),
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    async def _retrieve_docs(self, query: str) -> str:
        query_embedding = self.embed_model.encode(query).tolist()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._db_search, query_embedding)

    def _db_search(self, embedding):
        conn = self._get_db_connection()
        cur = conn.cursor()
        try:
            search_query = """
                SELECT content FROM loan_product_documents
                ORDER BY embedding <=> %s::vector
                LIMIT 3
            """
            cur.execute(search_query, (embedding,))
            rows = cur.fetchall()
            context = "\n\n".join([row[0] for row in rows])
            return context if context else "관련 문서 내용을 찾을 수 없습니다."
        finally:
            cur.close()
            conn.close()

    async def stream_answer(self, message: str, user_info: dict) -> AsyncGenerator[str, None]:
        name = user_info.get("name", "고객")
        income = user_info.get("income", 0)
        credit = user_info.get("creditScore", 0)
        session_id = user_info.get("session_id")

        if not session_id:
            raise ValueError("session_id is required")

        bot_chunks: list[str] = []

        try:
            context = await self._retrieve_docs(message)
            history = self._build_history_messages(session_id)

            self._save_chat_message(session_id, "USER", message)

            async for chunk in self.chain.astream(
                {
                    "name": name,
                    "income": income,
                    "credit": credit,
                    "context": context,
                    "history": history,
                    "message": message,
                }
            ):
                if chunk:
                    bot_chunks.append(chunk)
                    yield chunk

            bot_message = "".join(bot_chunks).strip()
            if bot_message:
                self._save_chat_message(session_id, "BOT", bot_message)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Error in stream_answer: {exc}")
            yield "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."

    def _get_user_profile(self, member_id: int):
        conn = self._get_db_connection()
        cur = conn.cursor()
        try:
            query = """
                SELECT
                    m.name,
                    ch.credit_score
                FROM member m
                LEFT JOIN (
                    SELECT DISTINCT ON (member_id)
                        member_id,
                        credit_score
                    FROM credit_history
                    ORDER BY member_id, credit_history_id DESC
                ) ch ON ch.member_id = m.member_id
                WHERE m.member_id = %s
            """
            cur.execute(query, (member_id,))
            row = cur.fetchone()

            if not row:
                raise ValueError("회원 정보를 찾을 수 없습니다.")

            name, credit = row
            return {
                "name": name,
                "credit": credit if credit is not None else "확인되지 않음",
            }
        finally:
            cur.close()
            conn.close()

chat_service = ChatService()
