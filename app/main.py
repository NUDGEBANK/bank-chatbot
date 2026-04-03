import os
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Dict, Any

#랭체인
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.prompts import PromptTemplate


load_dotenv()

app = FastAPI()


# 1-1. 프롬프트 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 NUDGEBANK 금융 상담 AI입니다. 사용자의 이름은 {name}입니다. 사용자의 연봉은 {income}만원, 신용점수는 {credit}점입니다. 이 정보를 바탕으로 사용자의 질문에 전문적이고 친절하게 답하세요."),
    ("user", "{message}")
])
# 1-2. LLM 모델 객체 생성
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500
)
# 1-3. 출력 파서 정의
output_parser = StrOutputParser()
# 2. 구성 요소 연결 ( 프롬프트->모델->출력파서 순으로 )
chain = prompt | llm | output_parser

# # 출력 방식: 스트리밍
# for chunk in chain.stream({"topic": "LLM"}):
#     print(chunk, end='')

class ChatRequest(BaseModel):
    user_id: str
    message: str
    user_info: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"message": "FastAPI AI 서버 실행 중"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    message = req.message
    user_info = req.user_info

    name = user_info.get("name", "고객")
    income = user_info.get("income", 0)
    credit = user_info.get("creditScore", 0)

    # 🔥 챗봇 로직
    # if "대출" in message:
    #     if income > 4000 and credit > 700:
    #         answer = f"{name}님, 대출 승인 가능성이 높습니다."
    #     else:
    #         answer = f"{name}님, 대출 심사가 필요합니다."
    # else:
    #     answer = f"{name}님, '{message}'에 대한 답변입니다."

    # 5. 체인 실행 (invoke 메서드 사용)
    # 딕셔너리 형태로 템플릿에 들어갈 변수들을 전달합니다.
    answer = await chain.ainvoke({
        "name": name,
        "income": income,
        "credit": credit,
        "message": req.message
    })

    


    return ChatResponse(answer=answer)
