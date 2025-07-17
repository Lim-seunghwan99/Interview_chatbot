# tools/interview_tools.py

import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


print("tools/interview_tools.py에서 임베딩 모델을 로드하는 중입니다...")
_embedding_model = SentenceTransformer("dragonkue/bge-m3-ko")

_chroma_client = chromadb.PersistentClient(path="my_interview_db")
_collection_name = "my_interviews_with_bge_m3"
_collection = _chroma_client.get_or_create_collection(
    name=_collection_name, metadata={"hnsw:space": "cosine"}
)
load_dotenv()
model = ChatOpenAI(model=os.getenv("LLM_MODEL_NAME"), temperature=0.7)


@tool
def find_similar_questions(topic: str, n: int = 3) -> List[str]:
    """
    주어진 주제(topic)와 가장 유사한 질문들을 벡터 DB에서 검색하여 반환합니다.

    Args:
        topic (str): 검색할 주제 키워드입니다.
        n (int): 검색할 유사 질문의 개수입니다.

    Returns:
        List[str]: 검색된 유사 질문 텍스트의 리스트를 반환합니다.
    """
    query_embedding = _embedding_model.encode(topic).tolist()
    results = _collection.query(query_embeddings=[query_embedding], n_results=n)
    retrieved_questions = results["documents"][0] if results.get("documents") else []
    return retrieved_questions


@tool
def evaluate_user_answer(question: str, user_answer: str) -> str:
    """
    주어진 면접 질문에 대한 사용자의 답변을 평가하고 건설적인 피드백을 제공합니다.
    """
    print(f"[{'답변 평가'}] 툴 실행")
    prompt = f"""
    당신은 친절하지만 핵심을 짚어주는 면접 코치입니다.
    아래의 질문과 답변을 분석하고, 답변의 좋은 점과 개선할 점을 구체적으로 피드백해주세요.
    특히, STAR 기법(Situation, Task, Action, Result)에 입각하여 답변이 구조적인지 평가해주세요.

    ### 면접 질문
    {question}

    ### 사용자의 답변
    {user_answer}

    ### 피드백 (아래 형식으로 작성)
    - **종합 평가**: (답변에 대한 전반적인 인상을 한두 문장으로 요약)
    - **좋은 점 (Good Points)**: (답변에서 칭찬할 만한 구체적인 부분)
    - **개선할 점 (Areas for Improvement)**: (답변을 더 좋게 만들기 위한 구체적인 조언)
    """
    feedback = model.invoke(prompt).content
    return feedback


@tool
def find_similar_qa_pairs(topic: str, n: int = 3) -> List[Dict[str, str]]:
    """
    주어진 주제와 유사한 <질문, 답변> 쌍을 벡터 DB에서 검색하여 반환합니다.
    질문은 문서(document)에서, 답변은 메타데이터(metadata)에서 가져옵니다.

    Args:
        topic (str): 검색할 주제 키워드입니다.
        n (int): 검색할 질문-답변 쌍의 개수입니다.

    Returns:
        List[Dict[str, str]]: {'question': ..., 'answer': ...} 형태의 딕셔너리 리스트.
    """
    query_embedding = _embedding_model.encode(topic).tolist()
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )
    qa_pairs = []
    if results.get("documents") and results.get("metadatas"):
        retrieved_docs = results["documents"][0]
        retrieved_metas = results["metadatas"][0]
        for doc, meta in zip(retrieved_docs, retrieved_metas):
            qa_pairs.append(
                {
                    "question": doc,
                    "answer": meta.get("answer", "저장된 답변 없음"),
                }
            )

    return qa_pairs
