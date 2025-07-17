# db/vector_db.py
import chromadb
from sentence_transformers import SentenceTransformer

try:
    _chroma_client = chromadb.PersistentClient(path="./chat_db")

    print("db/vector_db.py에서 임베딩 모델(dragonkue/bge-m3-ko)을 로드하는 중입니다...")
    _embedding_model = SentenceTransformer("dragonkue/bge-m3-ko")
    print("임베딩 모델 로드 완료.")

    _collection = _chroma_client.get_or_create_collection(
        name="chat_history_collection",
        metadata={"hnsw:space": "cosine"},
    )
    print("ChromaDB 컬렉션 준비 완료.")

except Exception as e:
    print(f"ChromaDB 또는 임베딩 모델 초기화 중 오류 발생: {e}")
    _chroma_client = None
    _embedding_model = None
    _collection = None


# 2. 채팅 기록 저장(임베딩) 함수
def add_chat_history_to_db(chatroom_id: str, chat_content: str):
    """주어진 채팅 내용을 임베딩하여 ChromaDB에 저장(또는 업데이트)합니다."""
    if not all([_collection, _embedding_model, chat_content]):
        print("DB, 모델 또는 내용이 준비되지 않아 저장을 건너뜁니다.")
        return

    print(f"'{chatroom_id}'의 채팅 기록을 임베딩하여 DB에 저장합니다...")
    embedding = _embedding_model.encode(chat_content).tolist()

    _collection.upsert(
        ids=[chatroom_id],
        embeddings=[embedding],
        documents=[chat_content],
        metadatas=[{"chatroom_id": chatroom_id}],
    )
    print(f"'{chatroom_id}' 저장 완료.")


# 3. 채팅 기록 조회 함수 수정 (ChromaDB 사용)
def get_chat_history_by_chatroom(chatroom_id: str) -> str | None:
    """ChromaDB에서 chatroom_id를 기준으로 채팅 기록(원본 텍스트)을 조회합니다."""
    if not _collection:
        print("DB가 준비되지 않아 조회할 수 없습니다.")
        return None

    print(f"ChromaDB에서 '{chatroom_id}' 채팅 기록 조회 시도...")

    try:
        result = _collection.get(ids=[chatroom_id], include=["documents"])
        if result and result.get("documents"):
            return result["documents"][0]
        else:
            return None

    except Exception as e:
        print(f"DB 조회 중 오류 발생: {e}")
        return None
