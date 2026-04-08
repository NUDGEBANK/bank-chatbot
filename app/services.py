import os
import asyncio
import psycopg2
from dotenv import load_dotenv
from typing import AsyncGenerator
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 추가된 임포트
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

class ChatService:
    def __init__(self):
        # 세션별 대화 기록을 저장할 딕셔너리 (메모리 저장소)
        self.store = {}

        # 1. 임베딩 모델 로드
        self.embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 2. 프롬프트 정의 (history 변수를 위한 MessagesPlaceholder 추가)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 NUDGEBANK 금융 상담 AI입니다.
사용자의 이름은 {name}님이며, 현재 정보는 다음과 같습니다:
- 연봉: {income}만원
- 신용점수: {credit}점
답변은 최대한 요점만 간단명료하게 작성하세요.
아래 제공된 [문서 내용]을 바탕으로 사용자의 질문에 전문적이고 친절하게 답변하세요.
만약 문서에 관련 내용이 없다면 아는 범위 내에서 답하되, 정확한 확인을 위해 약관 참고가 필요하다고 안내하세요.

[문서 내용]
{context}
""".strip()),
            MessagesPlaceholder(variable_name="history"), # 대화 기록 주입
            ("user", "{message}")
        ])
        
        # 3. LLM 모델 선언
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500,
            streaming=True,
        )
        
        # 4. 출력 파서
        self.output_parser = StrOutputParser()
        
        # 5. 기본 체인 연결(LCEL)
        self.chain = self.prompt | self.llm | self.output_parser

        # 6. RunnableWithMessageHistory로 체인 래핑
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="message", # 사용자 입력 키
            history_messages_key="history", # 프롬프트의 MessagesPlaceholder 이름
        )

    # 세션 ID에 맞는 대화 기록 객체를 반환하거나 새로 생성하는 함수
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def _get_db_connection(self):
        # DB 연결 및 pgvector 등록
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS")
        )
        register_vector(conn)
        return conn

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
            return context if context else "관련된 약관 내용을 찾을 수 없습니다."
        finally:
            cur.close()
            conn.close()

    async def stream_answer(self, message: str, user_info: dict) -> AsyncGenerator[str, None]:
        name = user_info.get("name", "고객")
        income = user_info.get("income", 0)
        credit = user_info.get("creditScore", 0)
        
        # user_info에서 고유 세션 ID 추출 (없으면 기본값 사용)
        # 실제 환경에서는 사용자 ID나 세션 토큰을 반드시 전달받아야 합니다.
        session_id = user_info.get("session_id", "default_session") 

        try:
            # 1. 리트리버 실행
            context = await self._retrieve_docs(message)

            # 2. 체인 실행 (스트리밍 및 history 자동 적용)
            async for chunk in self.chain_with_history.astream(
                {
                    "name": name,
                    "income": income,
                    "credit": credit,
                    "context": context,
                    "message": message
                },
                # config를 통해 현재 실행의 session_id를 전달
                config={"configurable": {"session_id": session_id}}
            ):
                if chunk:
                    yield chunk
                    
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Error in stream_answer: {e}")
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
                "credit": credit if credit is not None else "확인되지 않음"
            }
        finally:
            cur.close()
            conn.close()


# 싱글톤 패턴으로 서비스 인스턴스 생성
chat_service = ChatService()