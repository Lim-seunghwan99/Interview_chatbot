# services/persona_analyzer.py
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

_internal_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
    temperature=0.7,
)

# --- 6가지 페르소나 상세 정보 ---
HEXACO_PERSONA_DESCRIPTIONS = """
### 1. 별들의 궤도를 기록하는 천문학자
- HEXACO 기반 특징: 성실성(Conscientiousness)과 정직성-겸손(Honesty-Humility)이 높음.
- 대화에서 드러나는 모습: 체계적, 논리적, 목표 지향적 대화 선호.
### 2. 유쾌한 축제를 여는 꿀벌
- HEXACO 기반 특징: 외향성(eXtraversion)과 원만성(Agreeableness)이 높음.
- 대화에서 드러나는 모습: 활기찬 리액션, 대화 주도, 긍정적 분위기 메이커.
### 3. 상처 입은 새를 돌보는 정원사
- HEXACO 기반 특징: 정서성(Emotionality)과 원만성(Agreeableness)이 높음.
- 대화에서 드러나는 모습: 깊은 공감, 따뜻한 위로, 인내심 있는 경청.
### 4. 미지의 바다를 그리는 항해사
- HEXACO 기반 특징: 경험에 대한 개방성(Openness to Experience)이 높음.
- 대화에서 드러나는 모습: 창의적, 독창적, 상상력을 자극하는 질문.
### 5. 오래된 숲의 약속을 지키는 문지기
- HEXACO 기반 특징: 정직성-겸손(Honesty-Humility)이 매우 높음.
- 대화에서 드러나는 모습: 솔직, 직설적, 원칙 중시, 말과 행동의 일치.
### 6. 자신만의 길을 걷는 길들여지지 않는 늑대
- HEXACO 기반 특징: 원만성(Agreeableness)과 정서성(Emotionality)이 낮은 편.
- 대화에서 드러나는 모습: 뚜렷한 주관, 독립적, 논쟁을 두려워하지 않음.
"""


def analyze_persona_from_history(chat_history: str) -> dict | None:
    """대화 기록(chat_history)을 직접 받아 페르소나를 분석하고 결과를 딕셔너리로 반환합니다."""
    if not chat_history:
        print("분석할 대화 내용이 없습니다.")
        return None

    prompt = f"""
    당신은 HEXACO 성격 모델 기반의 대화 스타일 분석가입니다.
    아래 6가지 페르소나 설명과 사용자 대화 내용을 바탕으로, 이 대화에서 사용자의 페르소나를 분석해주세요.

    {HEXACO_PERSONA_DESCRIPTIONS}  
    ---
    ### 사용자 대화 내용
    {chat_history}
    ---
    ### 분석 요청
    위 대화 내용에서 사용자 'B'의 스타일에 가장 적합한 페르소나를 **단 하나만** 선택하고, 아래 형식에 맞춰 분석 결과를 작성해주세요.

    ### 분석 결과
    - **당신의 대화 페르소나**: 
    - **판단 근거**: 
    - **당신은 이런 점이 멋져요!**: 
    """

    try:
        response_text = _internal_llm.invoke(prompt).content

        lines = response_text.strip().split("\n- ")
        result = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().replace("*", "")
                if key == "당신의 대화 페르소나":
                    result["persona"] = value.strip()
                elif key == "판단 근거":
                    result["reasoning"] = value.strip()
                elif key == "당신은 이런 점이 멋져요!":
                    result["feedback"] = value.strip()

        return result if "persona" in result else None

    except Exception as e:
        print(f"페르소나 분석 중 오류 발생: {e}")
        return None
