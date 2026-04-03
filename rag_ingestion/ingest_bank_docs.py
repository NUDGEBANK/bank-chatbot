import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
# DB 접속 설정
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

def get_db_connection():
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
    register_vector(conn)
    return conn

def main():
    # 2. 모델 및 설정 준비
    print("임베딩 모델 로딩 중...")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    # PDF 파일 경로 (파일명 확인 필수!)
    pdf_filename = "약관 및 상품설명서.pdf"
    # 상대 경로: rag_ingestion/docs/약관 및 상품.pdf
    pdf_path = os.path.join("rag_ingestion", "docs", pdf_filename)

    if not os.path.exists(pdf_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        return

    # 3. LangChain을 이용한 PDF 로드 및 텍스트 분할
    print(f"'{pdf_filename}' 읽는 중...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 의미 단위로 700자씩 자르되, 문맥 유지를 위해 100자씩 겹침
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    split_docs = text_splitter.split_documents(pages)

    # 4. DB 연결 및 데이터 적재
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # pgvector 타입 등록
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)

        # --- [덮어쓰기 로직] ---
        # 현재 파일명으로 시작하는 기존 데이터를 모두 삭제
        cur.execute("DELETE FROM loan_product_documents WHERE doc_name LIKE %s", (f"{pdf_filename}%",))
        print(f"♻️ 기존 '{pdf_filename}' 관련 데이터를 삭제하고 새로 적재합니다.")
        # ----------------------------------

        print(f"총 {len(split_docs)}개의 청크 적재 시작...")
        
        # 임시 상품 ID (실제 DB 상품 테이블의 ID와 매칭 권장)
        loan_product_id = 1 

        for i, doc in enumerate(split_docs):
            content = doc.page_content
            # 메타데이터에서 원본 페이지 번호 추출 (0부터 시작하므로 +1)
            original_page = doc.metadata.get('page', 0) + 1
            full_doc_name = f"{pdf_filename} (P.{original_page})"
            
            # 임베딩 생성
            embedding_vector = model.encode(content).tolist()
            
            # DB INSERT (이미지 모델 구조와 100% 일치)
            insert_query = """
                INSERT INTO loan_product_documents 
                (loan_product_id, doc_name, chunk_index, content, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """
            cur.execute(insert_query, (
                loan_product_id, 
                full_doc_name, 
                i, # chunk_index
                content, 
                embedding_vector
            ))
            
            if (i + 1) % 5 == 0:
                print(f"  -> {i + 1}개 적재 진행 중...")

        conn.commit()
        print(f"\n✅ 성공: 모든 청크가 'nudgebank' DB에 적재되었습니다!")

    except Exception as e:
        conn.rollback()
        print(f"❌ 오류 발생: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()