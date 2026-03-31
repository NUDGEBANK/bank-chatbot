import os
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI()

@app.get("/")
def root():
    return {"message": "FastAPI AI 서버 실행 중"}