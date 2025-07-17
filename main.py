# main.py
from api.router import router as api_router
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from services.persona_analyzer import (
    analyze_persona_from_history,
)
from db.vector_db import get_chat_history_by_chatroom
from schemas import AnalysisResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="다목적 AI 어시스턴트 API")
app.include_router(api_router, prefix="/api")


@app.get("/", summary="API 서버 동작 확인")
def read_root():
    return {"message": "AI 어시스턴트 API 서버가 정상적으로 동작하고 있습니다."}


# 페르소나 분석
@app.post(
    "/analyze/chatroom/{chatroom_id}",
    response_model=AnalysisResponse,
    summary="채팅방 페르소나 분석",
    description="주어진 `chatroom_id`의 대화 내용을 분석하여 사용자의 페르소나, 판단 근거, 긍정적 피드백을 반환합니다.",
)
def analyze_chatroom_endpoint(chatroom_id: str):
    """
    채팅방 ID를 받아 해당 채팅방의 대화 페르소나 분석을 요청하는 엔드포인트입니다.
    """
    if not chatroom_id:
        raise HTTPException(status_code=400, detail="채팅방 ID가 필요합니다.")
    chat_history = get_chat_history_by_chatroom(chatroom_id)

    if not chat_history:
        raise HTTPException(
            status_code=404,
            detail=f"'{chatroom_id}'에 해당하는 채팅방을 찾을 수 없거나 대화 내용이 없습니다.",
        )
    analysis_result = analyze_persona_from_history(chat_history)

    if not analysis_result:
        raise HTTPException(
            status_code=500,
            detail="대화 내용 분석에 실패했습니다. 나중에 다시 시도해주세요.",
        )

    return analysis_result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 웹소켓 연결 성공 및 수락 완료")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"클라이언트로부터 받은 메시지: {data}")
            await websocket.send_text(f"서버가 받은 메시지: {data}")
    except WebSocketDisconnect:
        print("🔌 클라이언트 연결이 끊어졌습니다.")
    except Exception as e:
        print(f"🚨 웹소켓 에러 발생: {e}")


# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# uvicorn main:app --reload --port 8000
