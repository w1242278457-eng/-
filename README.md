# Campus Customer Service Multi-Agent

面向校园场景的智能客服多 Agent 系统，适用于招生咨询、教务问答、校园卡服务、宿舍报修、奖助学金说明、校园生活指引等业务。

## 项目定位

这个仓库当前实现的是一个基于 `LangGraph + FastAPI` 的多 Agent 后端。我们已经将原本偏金融的客服语义、知识库示例、工单类型和合规策略，调整为校园客服场景。

如果后续需要继续扩展到你提到的 `Flask + Streamlit` 架构，可以在现有 Agent 编排和接口层之上平滑迁移。

## 系统特点

- 多 Agent 协同：Supervisor 负责路由，Knowledge RAG 负责知识问答，Ticket Handler 负责工单受理，Compliance Checker 负责隐私与内容安全审查
- 状态编排：通过 `StateGraph` 串联各个 Agent，避免复杂咨询场景下的重复回复和路由失控
- 本地知识库：内置校园 FAQ 示例，可继续扩展为教务、宿管、后勤、奖助学金等结构化知识
- 工具协议：内置 MCP 风格工具注册与调用接口，便于对接校园卡、报修、教务、图书馆等外部系统
- 会话记忆：包含工作记忆、短期记忆和长期记忆，支持多轮对话
- 可观测性：接入 OpenTelemetry，便于追踪 Agent 调用链路

## 适合的校园业务

- 招生与报到咨询
- 选课、考试、成绩、学籍相关问答
- 宿舍报修与后勤服务
- 校园卡挂失、补办、充值咨询
- 奖学金、助学金、勤工助学政策说明
- 投诉建议与人工转办

## 目录结构

```text
python-impl/
├── agents/              # 多 Agent 核心逻辑
│   ├── supervisor.py
│   ├── intent_router.py
│   ├── knowledge_rag.py
│   ├── ticket_handler.py
│   └── compliance_checker.py
├── api/                 # FastAPI 接口
├── memory/              # 工作记忆 / 短期记忆 / 长期记忆
├── mcp/                 # 工具协议与默认工具
└── tracing/             # 链路追踪
```

## 快速启动

```bash
cd python-impl
pip install -r requirements.txt
python -m api.main
```

默认启动后接口地址为 `http://localhost:8000`。

## 已完成的校园化改造

- 将内置知识库样例从理财、退款、开户改为选课、宿舍报修、校园卡、奖助学金
- 将工单类型改为宿舍报修、校园卡、奖助学金、教务、投诉建议等校园业务
- 将合规规则改为校园隐私保护、成绩与学籍信息保护、危险指引拦截、越权承诺识别
- 将默认工具改为校园卡查询、校内知识检索、服务工单创建、紧急事件分级

## 后续可扩展方向

- 接入真实的教务系统、宿舍报修系统、校园卡系统
- 增加结构化表格检索与 Pandas 查询
- 增加 Web 搜索 API 作为外部工具
- 增加前端页面与流式输出
- 按学院、角色、校区做细粒度路由

## License

MIT
