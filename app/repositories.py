import uuid

from .db import get_db_connection


class ChatRepository:
    def load_session_messages(self, session_id: str) -> list[tuple[str, str]]:
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT sender_type, message_content
                FROM (
                    SELECT message_id, sender_type, message_content
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY message_id DESC
                    LIMIT 10 -- 최근 메세지 로드 제한
                ) AS recent_messages
                ORDER BY message_id ASC
                """,
                (session_id,),
            )
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()

    def search_past_conversations(
        self, member_id: int, current_session_id: str, query_embedding: list[float], limit: int = 3) -> str:
        #현재 세션을 제외한 해당 멤버의 모든 과거 대화 중 유사도가 높은 내역 검색
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT m.sender_type, m.message_content
                FROM chat_messages m
                JOIN chat_sessions s ON m.session_id = s.session_id
                WHERE s.member_id = %s 
                  AND m.session_id != %s 
                  AND m.embedding IS NOT NULL
                ORDER BY m.embedding <=> %s::vector
                LIMIT %s
                """,
                (member_id, current_session_id, query_embedding, limit),
            )
            rows = cur.fetchall()
            return "\n".join([f"[{'사용자' if r[0]=='USER' else 'NUDGEBOT'}] {r[1]}" for r in rows])
        finally:
            cur.close()
            conn.close()

    def prepare_chat_session(
        self, member_id: int, requested_session_id: str | None, title: str
    ) -> str:
        session_id = self.resolve_session_id(requested_session_id)
        conn = get_db_connection(register_vector_type=False)
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

    def save_chat_message(self, session_id: str, sender_type: str, message_content: str, embedding: list[float] | None = None) -> None:
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO chat_messages (session_id, sender_type, message_content, embedding, created_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (session_id, sender_type, message_content, embedding),
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

    def get_user_profile(self, member_id: int) -> dict:
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
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
                """,
                (member_id,),
            )
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

    def list_chat_sessions(self, member_id: int) -> list[dict]:
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT session_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE member_id = %s
                ORDER BY updated_at DESC NULLS LAST, created_at DESC, session_id DESC
                """,
                (member_id,),
            )
            rows = cur.fetchall()
            return [
                {
                    "session_id": str(session_id),
                    "title": title or "새 상담",
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
                for session_id, title, created_at, updated_at in rows
            ]
        finally:
            cur.close()
            conn.close()

    def get_chat_session_detail(self, member_id: int, session_id: str) -> dict:
        normalized_session_id = self.resolve_session_id(session_id)
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT session_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE session_id = %s AND member_id = %s
                """,
                (normalized_session_id, member_id),
            )
            session_row = cur.fetchone()
            if session_row is None:
                raise ValueError("chat session not found")

            cur.execute(
                """
                SELECT message_id, sender_type, message_content, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, message_id ASC
                """,
                (normalized_session_id,),
            )
            message_rows = cur.fetchall()

            session_id_value, title, created_at, updated_at = session_row
            return {
                "session_id": str(session_id_value),
                "title": title or "새 상담",
                "created_at": created_at,
                "updated_at": updated_at,
                "messages": [
                    {
                        "message_id": message_id,
                        "sender_type": sender_type,
                        "message_content": message_content,
                        "created_at": created_at,
                    }
                    for message_id, sender_type, message_content, created_at in message_rows
                ],
            }
        finally:
            cur.close()
            conn.close()

    def rename_chat_session(self, member_id: int, session_id: str, title: str) -> dict:
        normalized_session_id = self.resolve_session_id(session_id)
        normalized_title = " ".join(title.split()).strip()
        if not normalized_title:
            raise ValueError("title is required")

        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE chat_sessions
                SET title = %s, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = %s AND member_id = %s
                RETURNING session_id, title, created_at, updated_at
                """,
                (normalized_title[:255], normalized_session_id, member_id),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("chat session not found")

            conn.commit()
            session_id_value, updated_title, created_at, updated_at = row
            return {
                "session_id": str(session_id_value),
                "title": updated_title,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        finally:
            cur.close()
            conn.close()

    def delete_chat_session(self, member_id: int, session_id: str) -> None:
        normalized_session_id = self.resolve_session_id(session_id)
        conn = get_db_connection(register_vector_type=False)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT 1
                FROM chat_sessions
                WHERE session_id = %s AND member_id = %s
                """,
                (normalized_session_id, member_id),
            )
            if cur.fetchone() is None:
                raise ValueError("chat session not found")

            cur.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id = %s
                """,
                (normalized_session_id,),
            )
            cur.execute(
                """
                DELETE FROM chat_sessions
                WHERE session_id = %s AND member_id = %s
                """,
                (normalized_session_id, member_id),
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def resolve_session_id(requested_session_id: str | None) -> str:
        if not requested_session_id:
            return str(uuid.uuid4())
        return str(uuid.UUID(requested_session_id))


class VectorRepository:
    def search_documents(self, embedding: list[float]) -> str:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT content FROM loan_product_documents
                ORDER BY embedding <=> %s::vector
                LIMIT 3
                """,
                (embedding,),
            )
            rows = cur.fetchall()
            context = "\n\n".join([row[0] for row in rows])
            return context if context else "관련 문서 내용을 찾을 수 없습니다."
        finally:
            cur.close()
            conn.close()
