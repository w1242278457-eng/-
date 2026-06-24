# MultiAgent-Chatbot

> 一个基于 LangGraph 的智能客服多Agent系统

## ✨ 项目简介

这是我在学习多Agent系统时开发的一个智能客服项目，通过多个AI Agent的协作来模拟真实的客服对话场景。

## 🛠️ 技术栈

- **LangGraph** - 多Agent编排框架
- **FastAPI** - API接口
- **FAISS** - 向量数据库
- **Redis** - 缓存/会话管理
- **OpenTelemetry** - 链路追踪

## 🚀 快速开始

```bash
cd python-impl
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 配置API Key
python -m api.main
```

## 📁 项目结构

```
python-impl/
├── agents/          # Agent实现
│   ├── supervisor.py      # 编排中心
│   ├── intent_router.py   # 意图识别
│   ├── knowledge_rag.py   # 知识检索
│   ├── ticket_handler.py  # 工单处理
│   └── compliance_checker.py # 合规审查
├── memory/          # 记忆系统
├── mcp/             # 工具协议
├── tracing/         # 链路追踪
└── api/             # API接口
```

## 🧠 Agent架构

```
用户 → Supervisor → Intent Router → Knowledge RAG → Compliance → 回复
                          ↘ Ticket Handler ↗
```

## 📝 License

MIT

## 📫 联系

GitHub: @w1242278457
