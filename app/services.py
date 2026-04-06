import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class ChatService:
    def __init__(self):
        # 1. 프롬프트 정의
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 NUDGEBANK 금융 상담 AI입니다 "
                       "사용자의 이름은 {name}입니다. 사용자의 연봉은 {income}만원, "
                       "신용점수는 {credit}점입니다. 이 정보를 바탕으로 사용자의 질문에 전문적이고 친절하게 답하세요."),
            ("user", "{message}")
        ])
        
        # 2. LLM 모델 선언
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=500
        )
        
        # 3. 출력 파서 (응답 텍스트만 추출)
        self.output_parser = StrOutputParser()
        # 4. 체인 연결(LCEL)
        self.chain = self.prompt | self.llm | self.output_parser

    async def get_answer(self, message: str, user_info: dict) -> str:
        name = user_info.get("name", "고객")
        income = user_info.get("income", 0)
        credit = user_info.get("creditScore", 0)

        # 5. 체인 실행 (비동기)
        answer = await self.chain.ainvoke({
            "name": name,
            "income": income,
            "credit": credit,
            "message": message
        })
        return answer

# 싱글톤 패턴으로 서비스 인스턴스 생성
chat_service = ChatService()