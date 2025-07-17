# api/router.py
import io
import os
import tempfile
import shutil
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from core.agent import process_user_request
from tools.speech_to_text import SpeechToTextTool
from schemas import UserRequest, FinalResponse, ChatHistoryItem
from fastapi.responses import StreamingResponse
from tools.text_to_speech import TextToSpeechTool
from db.vector_db import add_chat_history_to_db


router = APIRouter()
stt_transcriber = SpeechToTextTool(whisper_model_name="base")
tts_synthesizer = TextToSpeechTool()


router.post(
    "/chatrooms/",
    status_code=202,
    summary="채팅 기록을 DB에 비동기로 추가",
    description="채팅 기록을 받아 백그라운드에서 임베딩하고 Vector DB에 저장합니다.",
)


#  대화가 어느 정도 쌓이거나, 대화 세션이 종료될 때 호출합니다. 매번 메시지를 보낼 때마다 전체 대화 로그를 임베딩하여 저장하는 것은 매우 비효율적.
async def add_chatroom_data(item: ChatHistoryItem, background_tasks: BackgroundTasks):
    """
    이 엔드포인트는 임베딩처럼 오래 걸릴 수 있는 작업을
    백그라운드로 보내고 즉시 응답을 반환합니다.
    """
    if not item.chatroom_id or not item.content:
        raise HTTPException(
            status_code=400, detail="chatroom_id와 content가 필요합니다."
        )

    background_tasks.add_task(
        add_chat_history_to_db, chatroom_id=item.chatroom_id, chat_content=item.content
    )

    return {
        "message": f"'{item.chatroom_id}'의 채팅 기록이 성공적으로 접수되었습니다. 백그라운드에서 처리됩니다."
    }


@router.post(
    "/process-voice/", response_model=FinalResponse, summary="음성 입력을 받아 처리"
)
async def handle_voice_input(audio_file: UploadFile = File(...)):
    """
    음성 파일을 받아 텍스트로 변환하고, AI 에이전트를 통해 최종 응답을 반환합니다.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(audio_file.filename)[1]
        ) as temp:
            shutil.copyfileobj(audio_file.file, temp)
            temp_path = temp.name

        transcribed_text = stt_transcriber(audio_path=temp_path)

        if not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="음성을 인식하지 못했습니다.")

        agent_response = await process_user_request(transcribed_text)

        return {
            "input_type": "voice",
            "transcribed_text": transcribed_text,
            "response": agent_response,
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        await audio_file.close()


@router.post("/process-text/", summary="텍스트 입력을 받아 처리")
async def handle_text_input(request: UserRequest) -> Dict[str, Any]:
    """
    텍스트 입력을 받아 AI 에이전트를 통해 최종 응답을 반환합니다.
    """
    user_text = request.user_text

    if not user_text:
        raise HTTPException(status_code=400, detail="'text' 필드가 필요합니다.")

    agent_response = await process_user_request(user_text)

    return {
        "input_type": "text",
        "original_text": user_text,
        "response": agent_response,
    }


@router.post("/process-tts/", summary="텍스트를 음성으로 변환")
async def handle_tts_input(request_data: Dict[str, str]):
    text_to_speak = request_data.get("text")

    if not text_to_speak:
        raise HTTPException(status_code=400, detail="'text' 필드가 필요합니다.")

    try:
        audio_bytes = tts_synthesizer(text=text_to_speak)
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 생성 중 오류 발생: {e}")
