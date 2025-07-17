# core/agent.py
from langchain_openai import ChatOpenAI
from typing import Any, Dict

from tools.agent_tools import (
    predict_recipient_reaction,
    advise_on_communication_style,
    get_mbti_communication_advice,
    refine_text_content,
)
from tools.interview_tools import (
    find_similar_questions,
    evaluate_user_answer,
    find_similar_qa_pairs,
)
import os
from dotenv import load_dotenv

load_dotenv()

available_tools = [
    predict_recipient_reaction,
    advise_on_communication_style,
    get_mbti_communication_advice,
    refine_text_content,
    evaluate_user_answer,
    find_similar_qa_pairs,
    find_similar_questions,
]
llm = ChatOpenAI(model=os.getenv("LLM_MODEL_NAME"), temperature=0.7)

llm_with_tools = llm.bind_tools(available_tools)
tool_map = {tool.name: tool for tool in available_tools}


async def process_user_request(user_text: str) -> Dict[str, Any]:
    """
    사용자 입력을 받아 적절한 툴을 실행하거나 LLM 답변을 반환합니다.
    이 함수가 에이전트의 핵심 두뇌 역할을 합니다.
    """
    ai_message = llm_with_tools.invoke(user_text)

    if not ai_message.tool_calls:
        return {"agent_name": "GeneralLLM", "response": ai_message.content}

    chosen_tool_call = ai_message.tool_calls[0]
    chosen_tool = tool_map.get(chosen_tool_call["name"])

    if not chosen_tool:
        return {"error": f"알 수 없는 도구 '{chosen_tool_call['name']}' 호출"}

    try:
        tool_args = chosen_tool_call["args"]
        result = chosen_tool.invoke(tool_args)
        return {
            "agent_name": chosen_tool_call["name"],
            "request_args": tool_args,
            "response": result,
        }
    except Exception as e:
        return {"error": f"툴 '{chosen_tool_call['name']}' 실행 중 오류: {e}"}
