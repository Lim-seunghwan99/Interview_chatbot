import chromadb
import json
from sentence_transformers import SentenceTransformer

print("모델을 로드하는 중입니다...")
model = SentenceTransformer("dragonkue/bge-m3-ko")
print("모델 로드 완료.")

client = chromadb.PersistentClient(path="my_interview_db")

try:
    with open("data.json", "r", encoding="utf-8") as f:
        json_data_list = json.load(f)
except FileNotFoundError:
    print(
        "❌ 에러: 'data.json' 파일을 찾을 수 없습니다. 스크립트와 같은 경로에 파일이 있는지 확인하세요."
    )
    exit()

processed_payloads = []
for item in json_data_list:
    new_item = {}
    for key, value in item.items():
        if isinstance(value, list):
            new_item[key] = ", ".join(map(str, value))
        else:
            new_item[key] = value
    processed_payloads.append(new_item)


collection_name = "my_interviews_with_bge_m3"

print(f"'{collection_name}' 컬렉션 확인 중...")
existing_collections = [c.name for c in client.list_collections()]
if collection_name in existing_collections:
    client.delete_collection(name=collection_name)
    print(f"기존 컬렉션 '{collection_name}'을(를) 초기화했습니다.")
else:
    print("기존 컬렉션이 없어 새로 생성합니다.")


collection = client.get_or_create_collection(
    name=collection_name, metadata={"hnsw:space": "cosine"}
)

texts_to_embed = [item["question"] for item in processed_payloads]

print(f"{len(texts_to_embed)}개의 텍스트를 임베딩하는 중입니다...")
vectors_to_upload = model.encode(texts_to_embed).tolist()
print("임베딩 완료.")

ids_to_upload = [str(idx) for idx, item in enumerate(processed_payloads)]
payloads_to_upload = processed_payloads


collection.add(
    ids=ids_to_upload,
    embeddings=vectors_to_upload,
    metadatas=payloads_to_upload,
    documents=texts_to_embed,
)

print(
    f"✅ 총 {len(payloads_to_upload)}개의 데이터 포인트를 '{collection_name}' 컬렉션에 성공적으로 저장했습니다."
)
