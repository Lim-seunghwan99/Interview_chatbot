# tools/speech_to_text.py

from typing import Type
from pydantic import BaseModel, Field
import os
import whisper


class SpeechToTextToolInput(BaseModel):
    """음성을 텍스트로 변환하는 도구의 입력 스키마"""

    audio_path: str = Field(description="변환할 오디오 파일의 경로 (로컬 파일 시스템).")


class SpeechToTextTool:
    """
    주어진 오디오 파일을 Whisper를 사용하여 텍스트로 변환하는 도구입니다.
    """

    name: str = "speech_to_text_transcriber"
    description: str = (
        "오디오 파일을 입력받아 해당 오디오의 내용을 텍스트로 정확하게 변환합니다."
    )
    args_schema: Type[BaseModel] = SpeechToTextToolInput

    def __init__(self, whisper_model_name: str = "base"):
        try:
            print(f"Whisper 모델 '{whisper_model_name}' 로드 중...")
            self.model = whisper.load_model(whisper_model_name)
            print(f"Whisper 모델 '{whisper_model_name}' 로드 완료.")
        except Exception as e:
            print(f"Whisper 모델 로드 중 오류 발생: {e}")
            self.model = None

    def _run(self, audio_path: str) -> str:
        """
        오디오 파일을 텍스트로 변환하는 실제 로직을 실행합니다.
        """
        if self.model is None:
            return "오디오 변환 서비스를 사용할 수 없습니다. Whisper 모델 로드에 실패했습니다."

        if not os.path.exists(audio_path):
            return f"오디오 파일 경로를 찾을 수 없습니다: {audio_path}"

        try:
            result = self.model.transcribe(audio_path)

            transcribed_text = result["text"]
            return transcribed_text
        except Exception as e:
            raise Exception(f"오디오 변환 중 예외 발생: {e}")

    def __call__(self, audio_path: str) -> str:
        return self._run(audio_path)
