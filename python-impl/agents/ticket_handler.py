"""
工单处理 Agent，负责校园服务工单的创建与查询。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call


class TicketStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    PENDING_REVIEW = "pending_review"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


TICKET_SYSTEM_PROMPT = """你是校园服务工单处理 Agent，负责处理需要受理、转办或跟进的请求。

你的职责：
1. 判断用户是否需要创建工单
2. 提取工单关键信息，如类型、优先级、摘要、详细描述、工单号
3. 对于已提供工单号的消息，可识别为查询工单

工单类型：
- dorm_repair：宿舍报修
- campus_card：校园卡挂失、补办、异常
- scholarship：奖助学金申请与材料问题
- academic_affairs：教务办理、选课异常、成绩申诉
- complaint：投诉建议
- counseling：心理支持或特殊关怀转介
- general：通用服务申请

优先级规则：
- urgent：安全事故、紧急求助、宿舍严重漏水漏电、疑似人身风险
- high：影响当日学习生活的故障、校园卡无法使用、临近截止时间的办理异常
- medium：常规服务申请与跟进
- low：普通咨询补充或非紧急建议

请以 JSON 返回，例如：
{
  "action": "create|query|update",
  "ticket_type": "dorm_repair|campus_card|...",
  "priority": "low|medium|high|urgent",
  "summary": "工单摘要",
  "details": "详细描述",
  "ticket_id": "可选，查询时返回"
}
"""


class TicketStore:
    """内存工单存储。"""

    def __init__(self):
        self._tickets: dict[str, dict] = {}

    def create(
        self, ticket_type: str, priority: str, summary: str, details: str, user_id: str
    ) -> dict:
        ticket_id = f"CAMP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        ticket = {
            "ticket_id": ticket_id,
            "type": ticket_type,
            "priority": priority,
            "status": TicketStatus.CREATED.value,
            "summary": summary,
            "details": details,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._tickets[ticket_id] = ticket
        return ticket

    def query(self, ticket_id: str) -> Optional[dict]:
        return self._tickets.get(ticket_id)


class TicketHandlerAgent:
    """工单处理 Agent。"""

    def __init__(self, llm: ChatOpenAI, ticket_store: Optional[TicketStore] = None):
        self.llm = llm
        self.ticket_store = ticket_store or TicketStore()

    @trace_agent_call("ticket_analyze")
    async def analyze_request(self, user_message: str) -> dict:
        messages = [
            SystemMessage(content=TICKET_SYSTEM_PROMPT),
            HumanMessage(content=f"用户消息: {user_message}"),
        ]
        response = await self.llm.ainvoke(messages)

        import json

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "action": "create",
                "ticket_type": "general",
                "priority": "medium",
                "summary": user_message[:100],
                "details": user_message,
            }

    @trace_agent_call("ticket_create")
    async def create_ticket(self, ticket_info: dict, user_id: str) -> str:
        ticket = self.ticket_store.create(
            ticket_type=ticket_info.get("ticket_type", "general"),
            priority=ticket_info.get("priority", "medium"),
            summary=ticket_info.get("summary", ""),
            details=ticket_info.get("details", ""),
            user_id=user_id,
        )

        priority_label = {
            "low": "低",
            "medium": "中",
            "high": "高",
            "urgent": "紧急",
        }.get(ticket["priority"], "中")

        return (
            "校园服务工单已创建。\n\n"
            f"工单号: {ticket['ticket_id']}\n"
            f"类型: {ticket['type']}\n"
            f"优先级: {priority_label}\n"
            f"摘要: {ticket['summary']}\n"
            f"创建时间: {ticket['created_at']}\n\n"
            "请保留工单号，后续可用于查询处理进度。"
        )

    @trace_agent_call("ticket_query")
    async def query_ticket(self, ticket_id: str) -> str:
        ticket = self.ticket_store.query(ticket_id)
        if not ticket:
            return f"未找到工单号 {ticket_id}，请确认输入是否正确。"

        status_label = {
            "created": "已创建",
            "processing": "处理中",
            "pending_review": "待审核",
            "resolved": "已处理",
            "closed": "已关闭",
            "escalated": "已升级",
        }.get(ticket["status"], ticket["status"])

        return (
            "工单查询结果：\n\n"
            f"工单号: {ticket['ticket_id']}\n"
            f"状态: {status_label}\n"
            f"类型: {ticket['type']}\n"
            f"摘要: {ticket['summary']}\n"
            f"创建时间: {ticket['created_at']}\n"
            f"更新时间: {ticket['updated_at']}"
        )

    @trace_agent_call("ticket_handler_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_id = state.get("user_id", "anonymous")

        if not messages:
            return state

        last_message = messages[-1].content
        ticket_info = await self.analyze_request(last_message)
        action = ticket_info.get("action", "create")

        if action == "query" and ticket_info.get("ticket_id"):
            result = await self.query_ticket(ticket_info["ticket_id"])
        else:
            result = await self.create_ticket(ticket_info, user_id)

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "ticket_handler": result,
            },
        }
