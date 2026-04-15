import os
import tempfile
from collections.abc import Callable
from datetime import datetime

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .db import get_db_connection


def _normalize_source_name(doc_name: str) -> str:
    marker = " (P."
    if marker in doc_name:
        return doc_name.split(marker, 1)[0]
    return doc_name


class RagAdminRepository:
    def list_documents(self) -> list[dict]:
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT loan_product_id, doc_name, created_at
                FROM loan_product_documents
                ORDER BY created_at DESC NULLS LAST, loan_product_id DESC, chunk_index ASC
                """
            )
            rows = cur.fetchall()
        finally:
            cur.close()
            conn.close()

        grouped: dict[int, dict] = {}
        for loan_product_id, doc_name, created_at in rows:
            product_id = int(loan_product_id)
            source_name = _normalize_source_name(doc_name or "")
            current = grouped.get(product_id)
            if current is None:
                grouped[product_id] = {
                    "loan_product_id": product_id,
                    "source_name": source_name,
                    "chunk_count": 1,
                    "created_at": created_at,
                    "updated_at": created_at,
                }
                continue

            current["chunk_count"] += 1
            if created_at and (
                current["updated_at"] is None or created_at > current["updated_at"]
            ):
                current["updated_at"] = created_at

        return list(grouped.values())

    def get_existing_document(self, loan_product_id: int) -> dict | None:
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT doc_name, COUNT(*), MIN(created_at), MAX(created_at)
                FROM loan_product_documents
                WHERE loan_product_id = %s
                GROUP BY doc_name
                ORDER BY MAX(created_at) DESC NULLS LAST, doc_name ASC
                """,
                (loan_product_id,),
            )
            rows = cur.fetchall()
        finally:
            cur.close()
            conn.close()

        if not rows:
            return None

        total_chunks = sum(int(chunk_count) for _, chunk_count, _, _ in rows)
        source_name = _normalize_source_name(rows[0][0] or "")
        created_at = min((row[2] for row in rows if row[2] is not None), default=None)
        updated_at = max((row[3] for row in rows if row[3] is not None), default=None)
        return {
            "loan_product_id": loan_product_id,
            "source_name": source_name,
            "chunk_count": total_chunks,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def get_next_product_id(self) -> int:
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT GREATEST(
                    COALESCE((SELECT MAX(loan_product_id) FROM loan_product_documents), 0),
                    COALESCE((SELECT MAX(loan_product_id) FROM loan_product), 0)
                ) + 1
                """
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 1
        finally:
            cur.close()
            conn.close()

    def replace_document(
        self,
        *,
        loan_product_id: int,
        source_name: str,
        split_docs: list,
        embed_text: Callable[[str], list[float]],
        log: Callable[[str], None],
    ) -> dict:
        conn = get_db_connection(register_vector_type=True)
        cur = conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                DELETE FROM loan_product_documents
                WHERE loan_product_id = %s
                """,
                (loan_product_id,),
            )
            deleted_count = int(cur.rowcount or 0)
            if deleted_count > 0:
                log(
                    f"기존 상품 ID {loan_product_id} 문서 {deleted_count}개 청크를 삭제하고 새 문서를 적재합니다."
                )
            else:
                log(f"상품 ID {loan_product_id}에 기존 문서가 없어 바로 적재를 시작합니다.")

            log(f"총 {len(split_docs)}개 청크 적재 시작")
            for index, doc in enumerate(split_docs):
                content = doc.page_content
                original_page = int(doc.metadata.get("page", 0)) + 1
                full_doc_name = f"{source_name} (P.{original_page})"
                embedding_vector = embed_text(content)

                cur.execute(
                    """
                    INSERT INTO loan_product_documents
                    (loan_product_id, doc_name, chunk_index, content, embedding, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    (loan_product_id, full_doc_name, index, content, embedding_vector),
                )

                if (index + 1) % 5 == 0 or index + 1 == len(split_docs):
                    log(f"  -> {index + 1}/{len(split_docs)} 청크 적재 완료")

            conn.commit()
            log("적재 완료")
            return self.get_existing_document(loan_product_id) or {
                "loan_product_id": loan_product_id,
                "source_name": source_name,
                "chunk_count": len(split_docs),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def delete_document(self, loan_product_id: int) -> int:
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                DELETE FROM loan_product_documents
                WHERE loan_product_id = %s
                """,
                (loan_product_id,),
            )
            deleted = int(cur.rowcount or 0)
            conn.commit()
            return deleted
        finally:
            cur.close()
            conn.close()


class RagAdminService:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.repository = RagAdminRepository()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
        )

    def list_documents(self) -> list[dict]:
        return self.repository.list_documents()

    def ingest_pdf(
        self,
        *,
        upload_file: UploadFile,
        requested_product_id: int | None,
        overwrite_confirmed: bool,
    ) -> dict:
        logs: list[str] = []

        def log(message: str) -> None:
            logs.append(message)
            print(f"[RAG ADMIN] {message}")

        if not upload_file.filename:
            raise ValueError("업로드할 파일 이름을 확인할 수 없습니다.")
        if not upload_file.filename.lower().endswith(".pdf"):
            raise ValueError("PDF 파일만 업로드할 수 있습니다.")

        assigned_product_id = (
            requested_product_id
            if requested_product_id is not None
            else self.repository.get_next_product_id()
        )
        if requested_product_id is None:
            log(f"상품 ID 미입력: 자동으로 {assigned_product_id}번을 배정합니다.")
        else:
            log(f"상품 ID {assigned_product_id}로 업로드를 진행합니다.")

        existing = self.repository.get_existing_document(assigned_product_id)
        if existing and not overwrite_confirmed:
            warning = (
                f"상품 ID {assigned_product_id}에는 이미 {existing['source_name']} 문서가 "
                f"{existing['chunk_count']}개 청크로 등록되어 있습니다. 덮어쓰기를 확인해 주세요."
            )
            log(warning)
            return {
                "status": "needs_confirmation",
                "message": warning,
                "assigned_product_id": assigned_product_id,
                "document": existing,
                "logs": logs,
            }

        suffix = os.path.splitext(upload_file.filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
            upload_file.file.seek(0)
            tmp_file.write(upload_file.file.read())

        try:
            log(f"파일 수신 완료: {upload_file.filename}")
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            log(f"PDF 로드 완료: {len(pages)}페이지")
            split_docs = self.text_splitter.split_documents(pages)
            log(f"청크 분할 완료: {len(split_docs)}개")

            document = self.repository.replace_document(
                loan_product_id=assigned_product_id,
                source_name=upload_file.filename,
                split_docs=split_docs,
                embed_text=lambda text: self.embed_model.encode(text).tolist(),
                log=log,
            )
            return {
                "status": "completed",
                "message": f"{upload_file.filename} 적재가 완료되었습니다.",
                "assigned_product_id": assigned_product_id,
                "document": document,
                "logs": logs,
            }
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            upload_file.file.close()

    def delete_document(self, loan_product_id: int) -> dict:
        logs: list[str] = []

        def log(message: str) -> None:
            logs.append(message)
            print(f"[RAG ADMIN] {message}")

        existing = self.repository.get_existing_document(loan_product_id)
        if existing is None:
            raise ValueError("삭제할 문서를 찾을 수 없습니다.")

        deleted_chunks = self.repository.delete_document(loan_product_id)
        log(
            f"상품 ID {loan_product_id}의 문서 {existing['source_name']} 삭제 완료 "
            f"({deleted_chunks}개 청크)"
        )
        return {
            "message": "문서를 삭제했습니다.",
            "deleted_chunks": deleted_chunks,
            "logs": logs,
        }
