"""
MCP 工具协议服务端，提供校园客服可调用的默认工具。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional


@dataclass
class ToolDefinition:
    """工具定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    category: str = "general"
    requires_auth: bool = False


@dataclass
class ToolCallResult:
    """工具调用结果。"""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MCPToolServer:
    """简单的 MCP 风格工具注册与调用服务。"""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._call_log: list[ToolCallResult] = []

    def register_tool(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        category: str = "general",
        requires_auth: bool = False,
    ) -> Callable:
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=func,
                category=category,
                requires_auth=requires_auth,
            )
            return func

        return decorator

    def list_tools(self, category: Optional[str] = None) -> list[dict]:
        tools = []
        for tool in self._tools.values():
            if category and tool.category != category:
                continue
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                    "category": tool.category,
                }
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        import time

        tool = self._tools.get(name)
        if tool is None:
            result = ToolCallResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' not found. Available: {list(self._tools.keys())}",
            )
            self._call_log.append(result)
            return result

        start = time.time()
        try:
            output = await tool.handler(**arguments)
            result = ToolCallResult(
                tool_name=name,
                success=True,
                result=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as exc:
            result = ToolCallResult(
                tool_name=name,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )

        self._call_log.append(result)
        return result

    async def handle_jsonrpc(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", 1)

        try:
            if method == "tools/list":
                result = self.list_tools(category=params.get("category"))
            elif method == "tools/call":
                call_result = await self.call_tool(
                    params.get("name", ""),
                    params.get("arguments", {}),
                )
                result = {
                    "success": call_result.success,
                    "result": call_result.result,
                    "error": call_result.error,
                }
            elif method == "ping":
                result = {"status": "ok"}
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": req_id,
                }

            return {"jsonrpc": "2.0", "result": result, "id": req_id}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(exc)},
                "id": req_id,
            }

    def get_call_log(self, last_n: int = 100) -> list[dict]:
        return [
            {
                "tool": item.tool_name,
                "success": item.success,
                "duration_ms": item.duration_ms,
                "timestamp": item.timestamp,
                "error": item.error,
            }
            for item in self._call_log[-last_n:]
        ]


def create_default_tools(server: MCPToolServer) -> MCPToolServer:
    """注册默认校园工具。"""

    @server.register(
        name="campus_card_query",
        description="查询校园卡状态、余额或挂失状态",
        input_schema={
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "学号"},
                "card_no": {"type": "string", "description": "校园卡号"},
            },
        },
        category="student_services",
    )
    async def campus_card_query(student_id: str = "", card_no: str = "") -> dict:
        return {
            "student_id": student_id or "2026001234",
            "card_no": card_no or "CARD-10001",
            "status": "active",
            "balance": 52.5,
            "loss_reported": False,
        }

    @server.register(
        name="knowledge_search",
        description="搜索校内知识库并返回相关文档片段",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 3},
            },
            "required": ["query"],
        },
        category="knowledge",
    )
    async def knowledge_search(query: str, top_k: int = 3) -> list[dict]:
        return [
            {
                "content": f"关于“{query}”的校内知识库片段",
                "source": "campus_faq.md",
                "score": 0.95,
            }
        ][:top_k]

    @server.register(
        name="service_ticket_create",
        description="创建校园服务工单",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                },
                "category": {"type": "string"},
            },
            "required": ["title", "description"],
        },
        category="ticket",
    )
    async def service_ticket_create(
        title: str, description: str, priority: str = "medium", category: str = "general"
    ) -> dict:
        import uuid

        return {
            "ticket_id": f"CAMP-{uuid.uuid4().hex[:8].upper()}",
            "title": title,
            "category": category,
            "status": "created",
            "priority": priority,
        }

    @server.register(
        name="emergency_check",
        description="对宿舍安全、紧急求助等场景进行分级判断",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "event": {"type": "string"},
                "severity_hint": {"type": "string"},
            },
            "required": ["user_id", "event"],
        },
        category="safety",
    )
    async def emergency_check(
        user_id: str, event: str, severity_hint: str = "normal"
    ) -> dict:
        risk_level = "low"
        if any(keyword in event for keyword in ("漏电", "受伤", "火情", "打架", "晕倒")):
            risk_level = "high"
        elif severity_hint in {"medium", "high"}:
            risk_level = severity_hint

        return {
            "user_id": user_id,
            "event": event,
            "risk_level": risk_level,
            "requires_manual_review": risk_level == "high",
        }

    return server
