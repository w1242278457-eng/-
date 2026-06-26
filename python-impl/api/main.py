"""
FastAPI 入口，提供校园客服系统的 REST API。
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.supervisor import create_supervisor_graph
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from memory.working_memory import WorkingMemory
from mcp.mcp_server import MCPToolServer, create_default_tools
from tracing.otel_config import AgentMetrics, init_tracer

load_dotenv()


working_memory = WorkingMemory()
short_term_memory = ShortTermMemory(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)
long_term_memory = LongTermMemory(
    index_path=os.getenv("FAISS_INDEX_PATH", "./vector_store/faiss_index")
)
mcp_server = create_default_tools(MCPToolServer())
metrics = AgentMetrics()
graph = None


def seed_campus_knowledge() -> None:
    """加载默认校园知识库样例。"""
    seed_docs = [
        {
            "content": (
                "本科生选课时间通常分为预选、正选和补退选三个阶段。学生需登录教务系统完成选课，"
                "如遇课程容量已满，可关注补退选阶段是否释放名额。"
            ),
            "source": "course_selection_faq.md",
        },
        {
            "content": (
                "宿舍报修可通过后勤服务平台提交，需填写楼栋、宿舍号、故障描述和联系电话。"
                "水电类故障一般 24 小时内响应，公共设施故障优先处理。"
            ),
            "source": "dorm_repair_policy.md",
        },
        {
            "content": (
                "校园卡遗失后应第一时间办理挂失，可通过校园卡中心、小程序或自助终端操作。"
                "补卡通常需要本人持有效证件办理，原卡余额可转入新卡。"
            ),
            "source": "campus_card_guide.md",
        },
        {
            "content": (
                "奖学金评定通常参考学业成绩、综合素质和校级规定。助学金申请需按学期提交材料，"
                "具体时间和条件以学生工作部门当学期通知为准。"
            ),
            "source": "scholarship_aid_notice.md",
        },
    ]

    for doc in seed_docs:
        long_term_memory.add_document(content=doc["content"], source=doc["source"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    global graph

    init_tracer(
        service_name=os.getenv("OTEL_SERVICE_NAME", "campus-cs-multi-agent"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    graph = create_supervisor_graph(
        working_memory=working_memory,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
    )
    seed_campus_knowledge()

    yield


app = FastAPI(
    title="校园智能客服多 Agent 系统",
    description="基于 LangGraph 的校园咨询、工单与合规协同服务系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    compliance_passed: bool


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """主聊天接口。"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    await short_term_memory.add_message(session_id, "user", request.message)

    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc

    final_response = result.get("final_response", "系统处理异常，请稍后重试")

    await short_term_memory.add_message(session_id, "assistant", final_response)

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        compliance_passed=result.get("compliance_passed", True),
    )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取对话历史。"""
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/api/tools")
async def list_tools():
    """列出可用工具。"""
    return {"tools": mcp_server.list_tools()}


@app.post("/api/tools/call")
async def call_tool(request: dict):
    """调用工具。"""
    result = await mcp_server.call_tool(
        name=request.get("name", ""),
        arguments=request.get("arguments", {}),
    )
    return {
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.get("/api/metrics")
async def get_metrics():
    """获取系统指标。"""
    return {
        "agent_metrics": metrics.get_summary(),
        "tool_call_log": mcp_server.get_call_log(last_n=20),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
