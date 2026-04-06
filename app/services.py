import os
import asyncio
import psycopg2
from dotenv import load_dotenv
from typing import AsyncGenerator
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class ChatService:
    def __init__(self):
        # 1. 임베딩 모델 로드 (데이터 적재 시와 동일한 모델)
        self.embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 2. 프롬프트 정의 (문서 내용 주입을 위한 {context} 추가)
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
            ("user", "{message}")
        ])
        
        # 3. LLM 모델 선언 (스트리밍 설정 유지)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500,
            streaming=True,
        )
        
        # 3. 출력 파서 (응답 텍스트만 추출)
        self.output_parser = StrOutputParser()
        # 4. 체인 연결(LCEL)
        self.chain = self.prompt | self.llm | self.output_parser

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
        # 질문과 유사한 문서 조각을 DB에서 검색 (동기 함수를 비동기처럼 실행)
        # 임베딩 생성 (CPU 작업이므로 run_in_executor 권장되나 단순 구현 위해 직접 실행)
        query_embedding = self.embed_model.encode(query).tolist()
        
        # DB 검색 (psycopg2는 동기 라이브러리이므로 블로킹 방지를 위해 별도 처리 권장)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._db_search, query_embedding)

    def _db_search(self, embedding):
        # 실제 DB 검색 로직 (동기)
        conn = self._get_db_connection()
        cur = conn.cursor()
        try:
            # 코사인 유사도 기반 상위 3개 청크 검색
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

        # 4. 리트리버 실행 (관련 지식 검색)
        try:
            context = await self._retrieve_docs(message)

            # 5. 체인 실행 (스트리밍)
            async for chunk in self.chain.astream({
                "name": name,
                "income": income,
                "credit": credit,
                "context": context,
                "message": message
            }):
                if chunk:
                    yield chunk
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Error in stream_answer: {e}")
            yield "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."

# 싱글톤 패턴으로 서비스 인스턴스 생성
chat_service = ChatService()