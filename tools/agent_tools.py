# tools/agent_tools.py
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

_internal_llm = ChatOpenAI(model=os.getenv("LLM_MODEL_NAME"), temperature=0.7)


# 상대방 생각/감정 예측 툴
@tool
def predict_recipient_reaction(
    recipient_description: str, situation_description: str
) -> str:
    """
    (영문 설명) Predicts how a person with specific traits might think or feel in a given situation, using the LLM's general psychological knowledge.
    (한글 번역) 특정 성향의 사람이 주어진 상황에서 어떻게 생각하거나 느낄지, LLM의 일반적인 심리학적 지식을 이용해 예측합니다.
    """
    print(
        f"[{'상대방 생각/감정 예측 (LLM Only)'}] 에이전트 실행 (대상: {recipient_description}, 상황: {situation_description})"
    )

    prompt = f"""
    당신은 사람의 행동과 심리를 깊이 이해하는 통찰력 있는 분석가입니다.
    당신의 방대한 지식을 활용하여, 아래 인물이 주어진 상황에서 어떻게 생각하고 느낄지 현실적으로 예측해주세요.

    ### 분석 대상
    - **인물 특징**: {recipient_description}
    - **처한 상황**: {situation_description}

    ### 예측 결과 (아래 형식으로 작성)
    - **예상되는 생각/감정**: (주요 감정과 생각을 2~3가지 핵심으로 요약)
    - **그렇게 생각하는 이유**: (인물의 특징과 상황을 심리학적 관점에서 연결하여 이유를 설명)
    """

    try:
        response = _internal_llm.invoke(prompt).content
        return response
    except Exception as e:
        print(f"상대방 생각 예측 중 오류 발생: {e}")
        return "죄송합니다, 현재 상대방의 생각을 예측하기 어렵습니다. 잠시 후 다시 시도해 주세요."


# 상황별 대화 조언 툴
@tool
def advise_on_communication_style(recipient_description: str, my_message: str) -> str:
    """
    (영문 설명) Checks if a message is appropriate for a specific recipient and provides advice on how to improve it, based on the LLM's communication expertise.
    (한글 번역) 특정 수신자에게 메시지를 보내도 괜찮은지 확인하고, LLM의 커뮤니케이션 전문 지식에 기반하여 더 나은 표현을 조언합니다.
    """
    print(
        f"[{'상황별 대화 조언 (LLM Only)'}] 에이전트 실행 (대상: {recipient_description}, 메시지: {my_message})"
    )

    prompt = f"""
    당신은 수많은 상황별 대화 경험을 가진 뛰어난 커뮤니케이션 코치입니다.
    당신의 전문성을 바탕으로, 사용자가 보내려는 메시지를 특정 대상에게 더 효과적으로 전달할 수 있도록 구체적으로 조언해주세요.

    ### 조언 요청 정보
    - **메시지 수신자**: {recipient_description}
    - **사용자가 보내려는 메시지**: "{my_message}"

    ### 조언 (아래 형식으로 작성)
    - **현재 메시지 분석**: (현재 메시지가 상대에게 어떻게 받아들여질 수 있는지 당신의 경험에 비추어 분석)
    - **수정 제안**: (더 나은 표현, 말투, 단어 등을 포함한 구체적인 메시지 수정안 제시)
    - **핵심 팁**: (이 상황에서 기억해야 할 가장 중요한 소통 팁 1가지)
    """

    try:
        response = _internal_llm.invoke(prompt).content
        return response
    except Exception as e:
        print(f"대화 조언 중 오류 발생: {e}")
        return "죄송합니다, 현재 대화 조언을 제공할 수 없습니다. 잠시 후 다시 시도해 주세요."


# 특정 mbti를 가진 사람이 어떻게 생각하는 지
@tool
def get_mbti_communication_advice(
    mbti_type: str,
    situation: str,
    my_message: str = "",
) -> str:
    """
    (영문 설명) Simulates how a person with a specific MBTI type would think in a given situation and provides communication advice. If a user's message is provided, it evaluates that message.
    (한글 번역) 특정 MBTI 유형의 사람이 주어진 상황에서 어떻게 생각할지 시뮬레이션하고 소통 조언을 제공합니다. 사용자의 메시지가 제공되면 해당 메시지를 평가합니다.
    """
    print(f"[{'MBTI 소통 조언'}] 에이전트 실행")
    prompt = f"""
    당신은 MBTI를 포함한 다양한 성격 이론과 사람들의 소통 방식에 매우 능통한 심리 및 커뮤니케이션 전문가입니다.
    아래에 제시된 MBTI 유형과 특정 상황을 바탕으로, 요청하는 내용을 분석하고 가장 현실적인 답변을 제공해주세요.

    - **대상 MBTI 유형**: {mbti_type}
    - **주어진 상황**: {situation}
    ---
    """
    if my_message:
        prompt += f"""
        아래는 제가 이 사람에게 보내려고 하는 메시지입니다.
        - **내가 보내려는 메시지**: "{my_message}"

        ### 요청 사항
        1.  이 메시지가 위 상황에서 {mbti_type} 유형의 사람에게 어떤 감정이나 생각을 유발할지 분석해주세요.
        2.  이 메시지가 소통에 효과적인지 평가하고, 더 나은 대안이 있다면 추천해주세요.

        ### 분석 결과 (아래 형식으로 작성)
        - **메시지 분석 및 예상 반응**: (메시지에 대한 상대방의 예상 생각/감정을 구체적으로 설명)
        - **소통 효과 평가**: (메시지가 긍정적인지, 부정적인지, 오해의 소지가 있는지 등)
        - **추천 메시지 및 소통 팁**: (관계를 해치지 않고 내 의도를 잘 전달할 수 있는 더 나은 표현 제안)
        """
    else:
        prompt += f"""
        ### 요청 사항
        1.  위 상황에서 {mbti_type} 유형의 사람은 어떤 생각과 감정을 느낄 가능성이 높은지 심층적으로 분석해주세요.
        2.  이 사람과 원활하게 소통하려면 어떤 점을 고려해야 할지 조언해주세요.

        ### 분석 결과 (아래 형식으로 작성)
        - **{mbti_type}의 예상 생각/감정**: (상황에 대한 내면의 사고 과정과 감정 상태를 구체적으로 설명)
        - **판단 근거**: (MBTI의 각 지표(E/I, N/S, T/F, P/J)를 바탕으로 왜 그렇게 생각하는지 설명)
        - **소통 팁**: (이 사람과 대화할 때 도움이 될 만한 실질적인 조언)
        """

    try:
        response = _internal_llm.invoke(prompt).content
        return response
    except Exception as e:
        print(f"MBTI 소통 조언 중 오류 발생: {e}")
        return "죄송합니다, 현재 요청을 수행할 수 없습니다."


# 글 다듬기 툴
@tool
def refine_text_content(text_to_refine: str, refinement_mode: str) -> str:
    """
    (영문 설명) Refines a given text based on a specified mode. Available modes are "부드럽게" (soften), "매끄럽게" (smoothen), "오타수정" (proofread), and "요약" (summarize).
    (한글 번역) 주어진 텍스트를 명시된 모드에 따라 다듬습니다. 사용 가능한 모드: "부드럽게", "매끄럽게", "오타수정", "요약".
    """
    print(f"[{'글 다듬기 (LLM Only)'}] 에이전트 실행 (모드: {refinement_mode})")

    base_prompt = f"당신은 글을 다듬는 전문가입니다. 아래의 원문을 주어진 모드에 맞게 수정해주세요.\n\n# 원문:\n{text_to_refine}\n\n# 수정 결과:"

    if refinement_mode == "부드럽게":
        prompt = f"당신은 상냥한 커뮤니케이션 전문가입니다. 아래 원문의 핵심 의미는 유지하되, 훨씬 더 정중하고 부드러운 말투로 수정해주세요. 상대방이 긍정적으로 받아들일 수 있도록 어조를 바꿔주세요.\n\n# 원문:\n{text_to_refine}\n\n# 부드럽게 수정한 결과:"
    elif refinement_mode == "매끄럽게":
        prompt = f"당신은 전문 편집자입니다. 아래 원문의 문장 구조를 더 자연스럽고 논리적으로 만들어주세요. 어색한 표현을 수정하고 문장의 흐름을 개선해주세요.\n\n# 원문:\n{text_to_refine}\n\n# 매끄럽게 수정한 결과:"
    elif refinement_mode == "오타수정":
        prompt = f"당신은 꼼꼼한 교정 전문가입니다. 아래 원문에서 맞춤법, 띄어쓰기, 문법 오류를 모두 찾아 수정해주세요. 내용이나 스타일은 절대 바꾸지 말고, 오직 오류만 수정해주세요.\n\n# 원문:\n{text_to_refine}\n\n# 오류만 수정한 결과:"
    elif refinement_mode == "요약":
        prompt = f"당신은 핵심을 꿰뚫는 분석가입니다. 아래 원문의 가장 중요한 내용을 간결하게 요약해주세요. 독자가 전체 글을 읽지 않아도 핵심을 파악할 수 있도록 해주세요.\n\n# 원문:\n{text_to_refine}\n\n# 핵심 요약:"
    else:
        prompt = base_prompt

    try:
        response = _internal_llm.invoke(prompt).content
        return response
    except Exception as e:
        print(f"글 다듬기 중 오류 발생: {e}")
        return "죄송합니다, 현재 글을 다듬을 수 없습니다. 잠시 후 다시 시도해 주세요."
