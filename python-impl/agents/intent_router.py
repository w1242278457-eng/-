"""
意图路由 Agent，识别校园客服场景下的用户意图。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call


class IntentCategory(str, Enum):
    """校园客服一级意图分类。"""

    ACADEMIC = "academic"
    CAMPUS_LIFE = "campus_life"
    LOGISTICS = "logistics"
    STUDENT_SERVICES = "student_services"
    COMPLAINT = "complaint"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """意图识别结果。"""

    primary_intent: IntentCategory
    secondary_intent: str
    confidence: float
    entities: dict[str, str]
    suggested_agent: str


INTENT_SYSTEM_PROMPT = """你是校园客服系统的意图识别 Agent。

请从以下维度分析用户意图：
1. 一级意图：academic, campus_life, logistics, student_services, complaint, emergency
2. 二级意图：例如 course_selection, exam_schedule, dorm_repair, campus_card_loss, scholarship, enrollment, complaint_feedback
3. 置信度：0.0-1.0
4. 关键实体：提取课程名、学院、楼栋、宿舍号、工单号、时间等信息
5. 建议路由：
   - knowledge_rag：政策问答、流程说明、常见咨询
   - ticket_handler：报修、挂失、投诉、人工转办、服务申请
   - compliance_checker：涉及隐私泄露、安全风险、危险指引的问题

请以 JSON 格式返回，例如：
{
  "primary_intent": "academic",
  "secondary_intent": "course_selection",
  "confidence": 0.96,
  "entities": {"semester": "2026 春季学期"},
  "suggested_agent": "knowledge_rag"
}

校园场景规则：
- 涉及宿舍故障、校园卡挂失、投诉建议、人工介入，优先路由到 ticket_handler
- 涉及选课、成绩、奖助学金、报到、图书馆规则，优先路由到 knowledge_rag
- 涉及学生隐私、危险行为、违规代办、敏感数据外泄，优先路由到 compliance_checker
"""


class IntentRouterAgent:
    """意图路由 Agent。"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @trace_agent_call("intent_router")
    async def classify(self, user_message: str) -> IntentResult:
        """对用户消息进行意图分类。"""
        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=f"用户消息: {user_message}"),
        ]
        response = await self.llm.ainvoke(messages)

        import json

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {
                "primary_intent": "unknown",
                "secondary_intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "suggested_agent": "knowledge_rag",
            }

        primary_intent = result.get("primary_intent", "unknown")
        try:
            primary = IntentCategory(primary_intent)
        except ValueError:
            primary = IntentCategory.UNKNOWN

        return IntentResult(
            primary_intent=primary,
            secondary_intent=result.get("secondary_intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            entities=result.get("entities", {}),
            suggested_agent=result.get("suggested_agent", "knowledge_rag"),
        )

    @trace_agent_call("intent_router_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """作为图节点处理状态。"""
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1].content if messages else ""
        intent_result = await self.classify(last_message)

        return {
            **state,
            "intent": intent_result.suggested_agent,
            "sub_results": {
                **state.get("sub_results", {}),
                "intent_router": {
                    "primary": intent_result.primary_intent.value,
                    "secondary": intent_result.secondary_intent,
                    "confidence": intent_result.confidence,
                    "entities": intent_result.entities,
                },
            },
        }
