# tools/text_to_speech.py
import io
from gtts import gTTS


class TextToSpeechTool:
    def __call__(self, text: str, lang: str = "ko") -> bytes:
        """
        주어진 텍스트를 음성 데이터(bytes)로 변환하여 메모리에서 바로 반환합니다.
        """
        if not text.strip():
            raise ValueError("음성으로 변환할 텍스트가 없습니다.")

        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        return mp3_fp.read()
