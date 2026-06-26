"""
知识检索 Agent，面向校园 FAQ 与政策问答。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from memory.long_term import LongTermMemory
from tracing.otel_config import trace_agent_call


RAG_SYSTEM_PROMPT = """你是校园知识库问答 Agent，负责根据检索到的文档回答用户问题。

回答规则：
1. 严格依据检索到的文档内容回答，不要编造校规、政策或办理结果
2. 如果文档中没有足够信息，要明确说明，并建议联系对应部门老师或服务窗口
3. 回答要清晰、简洁，适合校园客服场景
4. 涉及奖助学金、成绩、学籍、选课时间等可能变动的信息时，提醒用户以最新校内通知为准
5. 在回答末尾注明引用的文档来源
"""

QUERY_REWRITE_PROMPT = """请将用户的口语化问题改写成更适合知识库检索的查询语句。
保留核心语义，补充校园业务关键词，只返回改写后的查询。

用户原始问题: {query}
"""


class KnowledgeRAGAgent:
    """知识检索 Agent。"""

    def __init__(self, llm: ChatOpenAI, long_term_memory: Optional[LongTermMemory] = None):
        self.llm = llm
        self.long_term_memory = long_term_memory or LongTermMemory()

    @trace_agent_call("rag_query_rewrite")
    async def rewrite_query(self, original_query: str) -> str:
        messages = [HumanMessage(content=QUERY_REWRITE_PROMPT.format(query=original_query))]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    @trace_agent_call("rag_retrieve")
    async def retrieve_documents(self, query: str, top_k: int = 5) -> list[dict]:
        return self.long_term_memory.search(query, top_k=top_k)

    @trace_agent_call("rag_rerank")
    async def rerank_documents(
        self, query: str, documents: list[dict], top_k: int = 3
    ) -> list[dict]:
        if not documents:
            return []

        doc_summaries = "\n".join(
            f"[{i}] {doc.get('content', '')[:200]}" for i, doc in enumerate(documents)
        )
        messages = [
            SystemMessage(content="你是校园知识库文档排序助手。"),
            HumanMessage(
                content=(
                    f"用户查询: {query}\n\n"
                    f"候选文档:\n{doc_summaries}\n\n"
                    f"请返回最相关的 {top_k} 个文档索引，用逗号分隔，例如 0,2,4"
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)

        try:
            indices = [int(item.strip()) for item in response.content.split(",")]
            reranked = [documents[i] for i in indices if i < len(documents)]
        except (ValueError, IndexError):
            reranked = documents[:top_k]

        return reranked

    @trace_agent_call("rag_generate")
    async def generate_answer(self, query: str, context_docs: list[dict]) -> str:
        if not context_docs:
            return (
                "抱歉，当前知识库中没有找到与这个校园问题直接相关的信息。"
                "建议联系对应学院、教务处、学生工作部门或后勤服务中心确认。"
            )

        context = "\n\n---\n\n".join(
            f"来源: {doc.get('source', 'unknown')}\n内容: {doc.get('content', '')}"
            for doc in context_docs
        )

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(
                content=f"用户问题: {query}\n\n检索到的参考文档:\n{context}"
            ),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content

    @trace_agent_call("knowledge_rag_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return state

        original_query = messages[-1].content
        rewritten_query = await self.rewrite_query(original_query)
        raw_docs = await self.retrieve_documents(rewritten_query, top_k=5)
        reranked_docs = await self.rerank_documents(rewritten_query, raw_docs, top_k=3)
        answer = await self.generate_answer(original_query, reranked_docs)

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "knowledge_rag": answer,
            },
        }
