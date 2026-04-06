import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class ChatService:
    def __init__(self):
        #임베딩 모델 로드 (ingest_bank_docs.py와 동일한 모델)
        self.embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

        # 1. 프롬프트 정의
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 NUDGEBANK 금융 상담 AI입니다. "
                       "답변은 최대한 간략하고 명확하게 하세요. "
                       "사용자의 이름은 {name}님이며, 연봉 {income}만원, 신용점수 {credit}점입니다. "
                       "아래 제공된 [문서 내용]을 바탕으로 사용자의 질문에 답변하세요. "
                       "만약 문서에 관련 내용이 없다면, 아는 범위 내에서 답변하되 정확한 확인은 약관 참고가 필요하다고 안내하세요.\n\n"
                       "[문서 내용]\n{context}"),
            ("user", "{message}")
        ])
        
        # 2. LLM 모델 선언
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500
        )
        
        # 3. 출력 파서 (응답 텍스트만 추출)
        self.output_parser = StrOutputParser()
        # 4. 체인 연결(LCEL)
        self.chain = self.prompt | self.llm | self.output_parser
    
    def _get_db_connection(self):
        """DB 연결 및 pgvector 등록"""
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
            """질문과 유사한 문서 조각을 DB에서 검색 (Retriever)"""
            # 질문을 벡터로 변환
            query_embedding = self.embed_model.encode(query).tolist()
            
            conn = self._get_db_connection()
            cur = conn.cursor()
            try:
                # 코사인 유사도(<=>) 기반 상위 3개 검색
                search_query = """
                    SELECT content FROM loan_product_documents 
                    ORDER BY embedding <=> %s::vector 
                    LIMIT 3
                """
                cur.execute(search_query, (query_embedding,))
                rows = cur.fetchall()
                
                # 검색된 결과 병합
                context = "\n\n".join([row[0] for row in rows])
                return context if context else "관련된 상품 약관 내용을 찾을 수 없습니다."
            finally:
                cur.close()
                conn.close()

    async def get_answer(self, message: str, user_info: dict) -> str:
        name = user_info.get("name", "고객")
        income = user_info.get("income", 0)
        credit = user_info.get("creditScore", 0)

        # 3. 리트리버 실행: 관련 지식 찾아오기
        context = await self._retrieve_docs(message)

        # 4. 체인 실행 (검색된 context 주입)
        answer = await self.chain.ainvoke({
            "name": name,
            "income": income,
            "credit": credit,
            "context": context,
            "message": message
        })
        return answer

chat_service = ChatService()