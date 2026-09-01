# Business Agent Harness

> 面向业务 Agent 应用的能力复用与运行框架。

Business Agent Harness 基于 Python、LangGraph 和 LangChain，为业务 Agent 提供可复用的运行能力。它将 Agent Loop、工具、权限、上下文、记忆和任务管理与业务逻辑解耦，支持从单 Agent 逐步扩展到知识检索和多 Agent 协作。`Knowledge Assistant` 用于验证文件分析、知识检索、报告生成和任务管理闭环。

当前状态：**第一阶段 M1-M5、第二阶段 RAG 最小闭环和第三阶段 Multi-Agent M7-M8 核心能力已完成；生产化能力仍处于后续规划阶段。**

详细实施路线请参阅：[通用 Agent 项目架构与实现计划](./plan/implementation-plan.md)

## 第一阶段项目亮点

第一阶段完成了一套面向业务 Agent 的单 Agent 执行闭环，覆盖请求处理、模型调用、工具执行和状态恢复，并通过 `Knowledge Assistant` 完成集成验证。

- **可复用 Agent Harness**：基于 LangGraph 封装 Agent Loop、工具、权限、上下文、记忆和任务能力，通过 `AgentProfile` 与业务逻辑解耦。
- **用户与会话隔离**：不同用户独立运行并隔离工作区、记忆、任务和产物；不同会话可并发执行，同一会话避免重复运行。
- **长会话治理**：自动压缩上下文，大型工具结果转存为 Artifact，Skill 和 Memory 按需加载。
- **可靠模型调用**：统一控制模型并发、重试和备用模型切换，并记录完整的调用事件。
- **任务与 MCP 扩展**：支持任务生命周期和并发认领，将外部 MCP 工具统一接入权限与故障隔离机制。
- **业务闭环验证**：`Knowledge Assistant` 已串联文件读取、计算、报告审批与生成、任务跟踪等典型流程。

## 第二阶段项目亮点

第二阶段在第一阶段 Harness 上完成了可关闭、可替换的 RAG 知识检索闭环，覆盖文档入库、授权检索、来源引用和增量更新，并通过 `Knowledge Assistant` 完成集成验证。

- **多格式文档入库**：支持 DOCX/Markdown/TXT、标题和表格提取、Frontmatter、稳定切分、批量 Embedding、增量更新和重建索引。
- **增量索引治理**：基于内容 Hash、Embedding 模型、向量维度、Splitter 版本和 Chunk ID 判断是否需要更新，避免重复处理未变化的文档。
- **用户与知识隔离**：公共知识与当前用户私有知识统一召回，用户 Scope 由 Runtime 绑定，模型不能传入或覆盖身份。
- **结构化目录查询**：通过 `document_catalog` 支持文档数量、枚举和精确元数据筛选，避免从有限的 Top-K 结果中错误统计全量文档。
- **真实来源引用**：检索结果经过排序、去重、阈值和上下文预算控制，并生成可定位到来源、章节和 Chunk 的 `[S1]` 引用。
- **可复用技术分层**：业务 Tool 只调用通用 RAG Pipeline，Embedding 与 PostgreSQL/pgvector 适配保留在 `services/rag/`，并支持关闭 RAG 后继续运行第一阶段单 Agent 能力。

## 第三阶段项目亮点：M7-M8 Multi-Agent

第三阶段在单 Agent 和 RAG 闭环之上完成了进程内 Multi-Agent 协作闭环，将通用的 Multi-Agent Harness 能力与 `Knowledge Assistant` 的 Business 能力分层，并通过业务场景完成任务拆分、并行执行、证据分析、结果审核和有限修订的集成验证。

- **Harness 能力——子任务契约与隔离运行时**：为每个子任务维护独立的 `task_id`、`team_run_id`、子线程、上下文、工具白名单、最大深度和执行预算，主会话状态不会自动透传。
- **Harness 能力——进程内 Team 通信协议**：MessageBus 提供 `send`、`publish`、`subscribe`、`close` 和 `drain`；Team Protocols 统一定义任务请求、接受、结果、失败、重试、审核、修订和 ACK 消息，并支持消息去重与可控关闭。
- **Harness 能力——并行与任务级重试**：无依赖任务通过 `delegate_many` 并行执行；失败时只重试当前子任务，成功的兄弟任务不会重跑，并使用最大尝试次数、指数退避和随机抖动。
- **Harness 能力——安全边界与失败隔离**：默认最大委派深度为一层，子 Agent 不能继续创建 Subagent；失败、超时、拒绝和联网不可用会转换为结构化结果，不会拖垮 Lead。
- **Harness 能力——可观测性与权限执行**：日志保留 `team_run_id`、角色、父任务、尝试次数、最后错误和工具白名单；子任务继续复用统一 Permission Pipeline。
- **Business 能力——Lead 任务拆分**：`Knowledge Assistant` 的 Lead 通过 `delegate_research`、`delegate_analysis` 和 `request_review` 组织 Researcher、Analyst 与 Reviewer，负责业务任务拆解和最终汇总。
- **Business 能力——Researcher 专项检索**：按业务场景划分 `catalog_researcher`、`rag_researcher` 和 `web_researcher`，分别面向文档目录、本地知识和公开互联网资料。
- **Business 能力——检索模式与证据分流**：遵循 `rag`、`web` 和 `hybrid` 策略；Hybrid 拆分为独立的本地 RAG 与 Web 任务，私有证据不会传给 Web Researcher。
- **Business 能力——Analyst 证据分析**：Analyst 只处理 Lead 传入的必要证据，支持比较、归类和受限计算，输出 findings、calculations、conclusions、citations 和 uncertainties。
- **Business 能力——Reviewer 独立审核**：Reviewer 检查事实支持、Citation、计算、证据冲突、推断边界和权限约束，输出 `approved`、`needs_revision` 或 `rejected`。
- **Business 能力——有限审核闭环**：Reviewer 退回结果后，Lead 可重新执行分析；运行时最多允许初审加两次修订，超过上限返回结构化失败原因。

M7-M8 默认只使用进程内资源，不依赖 Redis、后台 Worker 或公网；联网 Researcher 是否可用取决于运行时是否配置了 Web Search MCP Tool。

## 系统架构

### 总体架构

```mermaid
flowchart LR
    cli["CLI Client"] --> api["FastAPI Agent Server"]
    api --> conversation["Conversation Service"]
    conversation --> runtime["User Runtime Registry"]
    conversation --> conversationDb[("SQLite Conversation Index")]
    runtime --> agentLoop["LangGraph Agent Loop"]
    agentLoop --> checkpointDb[("SQLite LangGraph Checkpoint")]
    agentLoop --> permission["Permission Pipeline and Hooks"]
    permission --> tools["Business and Harness Tools"]
    tools --> userStores[("Per-user Tasks, Memory, Artifacts")]
    tools --> documentSearch["document_search"]
    documentSearch --> ragPipeline["RAG Pipeline"]
    ragPipeline --> vectorDb[("PostgreSQL + pgvector")]
    agentLoop --> teamCoordinator["TeamCoordinator / Lead"]
    teamCoordinator --> messageBus["In-memory MessageBus"]
    teamCoordinator --> researcher["Researcher"]
    teamCoordinator --> analyst["Analyst"]
    teamCoordinator --> reviewer["Reviewer"]
    indexer["CLI Indexer"] --> ingestion["Load, Split, Embed"]
    ingestion --> vectorDb
    agentLoop --> modelGateway["Model Gateway"]
    modelGateway -.-> modelProvider["Primary and Fallback Models"]
    agentLoop --> mcpAdapter["MCP Adapter"]
    mcpAdapter -.-> mcpServers["Configured MCP Servers"]
```

### 技术选型

详细的总体架构与技术选型映射见[实施计划](./plan/implementation-plan.md#61-总体架构与技术选型映射)。

### 核心分层

```text
src/
├── harness/                         # 通用 Agent Harness
│   ├── agent_loop.py                # AgentLoop 对外门面
│   ├── graph.py                     # LangGraph 节点和路由
│   ├── messages.py                  # 消息和 Tool 契约
│   ├── permissions.py               # Permission Pipeline
│   ├── tool_use.py                  # Tool 注册和执行
│   ├── conversation.py              # Conversation 和 Run 生命周期
│   └── capabilities/                # Todo、Skill、Memory、Task、RAG、Agent Teams 等
│       └── agent_teams/             # MessageBus、Team Protocols、调度和隔离
├── business/
│   └── knowledge_assistant/         # 当前业务 Agent
│       ├── profile.py               # 业务能力装配
│       ├── permission_rules.py      # 业务权限规则
│       ├── tools/                   # Calculator、File Reader、Report Writer、Document Search
│       ├── skills/                  # 内置 Skill
│       └── agent_teams/             # Lead、Researcher、Analyst、Reviewer
│           ├── lead.py              # 委派工具和审核入口
│           ├── researcher.py        # Catalog/RAG/Web Researcher 定义
│           ├── analyst.py           # 分析输入输出契约
│           └── reviewer.py          # 审核输入输出契约
├── services/                        # 模型、存储、MCP、日志和 RAG 技术实现
└── entrypoints/                     # CLI、API、Bootstrap 和 Indexer 入口

tests/
├── unit/                            # 单元测试
└── integration/                    # API、Checkpoint 和业务闭环测试
```

### 运行时职责

- `ConversationService` 管理会话所有权、Run 状态、并发控制和权限恢复。
- `UserRuntimeRegistry` 按用户创建并缓存 Runtime，绑定用户级目录和 Store。
- `AgentLoop` 负责模型、权限、工具和终止路由，不直接处理具体业务。
- `ModelGateway` 是共享的模型调用入口，负责重试、并发和降级。
- `Tool and Capability Layer` 执行本地工具、Task、Memory、Skill 和 MCP 工具。
- `RAG Pipeline` 负责有界检索和引用；FastEmbed 与 pgvector 负责向量化和持久化召回。
- SQLite Checkpoint 保存 Graph 状态；用户级 Store 保存 Task、Memory 和 Artifact。

### 持久化目录

```text
.agent/conversations.sqlite3       # 会话索引
users/<user_id>/
├── workspace/                     # 用户工作文件
├── knowledge/                     # 用户私有知识
├── skills/                        # 用户私有 Skill
├── memory/                        # 用户跨会话 Memory
├── checkpoints.sqlite3            # 用户 Checkpoint
└── tasks.sqlite3                 # 用户 Task
artifacts/<user_id>/               # 用户生成的报告和文件产物
```

## 快速启动

### 环境要求

- Python 3.12
- `uv`
- 一个已配置的模型 Provider

### 安装依赖

```bash
uv sync
cp .env.example .env
```

编辑 `.env`，至少配置：

```dotenv
AGENT_MODEL__PROVIDER=anthropic
AGENT_MODEL__MODEL_ID=your-model-id
AGENT_MODEL__API_KEY=your-api-key
```

### 启动 Agent Server

终端一：

```bash
uv run agent serve
```

默认监听 `127.0.0.1:8000`。也可以指定地址和端口：

```bash
uv run agent serve --host 127.0.0.1 --port 8000
```

### 启动 CLI

终端二：

```bash
uv run agent chat --user alice
```

CLI 内置命令：

```text
/new                         新建 Conversation
/list                        查看 Conversation
/switch <ID>                 切换 Conversation
/delete <ID>                 删除 Conversation
/cancel <ID>                 取消运行中的 Run
/usage                       查看 Token 用量
/model [模型名]              查看或切换当前 CLI 会话的主模型
/search-mode auto|rag|web|hybrid 设置当前会话的检索模式
/skills                      查看可用 Skill
/mcp                         查看 MCP Server 和工具
/test tool|skill|mcp ...     强制测试指定能力
/exit                        退出
```

#### `/model` 模型切换说明

`/model` 查看服务端当前配置的主模型和 fallback 模型；`/model <模型名>` 将该模型设为当前
CLI 会话后续消息的首选模型。模型必须已在服务端配置，切换不会影响其他用户或服务端全局主模型。

```text
/model
/model deepseek/deepseek-v4-flash
```

#### `/search-mode` 检索模式说明

检索模式：`auto` 自动路由；`rag` 仅本地 RAG；`web` 仅联网搜索；`hybrid` 同时使用两者。
其中 `web` 和 `hybrid` 需要通过 MCP 配置联网搜索 Tool。

CLI 测试联网搜索：

```text
/search-mode web
最新黄金价格是多少？
```

未配置联网搜索 Tool 时，Agent 不会调用本地 RAG 或编造实时数据，而会明确提示联网能力不可用。

### 启用 RAG

启动本地 PostgreSQL 16 + pgvector：

```bash
docker compose -f compose.rag.yaml up -d
```

在 `.env` 中启用并配置 RAG：

```dotenv
AGENT_RAG__ENABLED=true
AGENT_RAG__DATABASE_URL=postgresql+psycopg://agent:agent@127.0.0.1:5432/agent
AGENT_RAG__COLLECTION_NAME=knowledge_assistant
AGENT_RAG__EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
AGENT_RAG__EMBEDDING_DIMENSION=512
```

索引公共资料和用户私有资料：

```bash
uv run agent index \
  --source src/business/knowledge_assistant/knowledge \
  --scope public

uv run agent index \
  --source users/alice/knowledge \
  --scope user \
  --user alice
```

第一次索引会下载 FastEmbed 模型。索引完成并重启 Agent Server 后，Knowledge Assistant
会注册只读 `document_search` Tool；用户身份由 Runtime 绑定，不接受模型传入的 `user_id`。
关闭 `AGENT_RAG__ENABLED` 后不会注册该 Tool，第一阶段能力可独立运行。

### 添加 Skill

内置 Skill 放在：

```text
src/business/knowledge_assistant/skills/<skill-name>/SKILL.md
```

用户私有 Skill 放在：

```text
users/<user_id>/skills/<skill-name>/SKILL.md
```

Skill 必须包含 `name`、`description` 和非空正文：

```markdown
---
name: data-analysis
description: Analyze data and explain evidence behind conclusions.
---

# Data Analysis

1. Inspect only authorized materials.
2. Separate facts from assumptions.
3. Explain the calculation and conclusion.
```

新增 Skill 后需要重启 Agent Server，或重新创建对应用户 Runtime。可以使用以下命令检查：

```text
/skills
/test skill data-analysis 请使用这个 Skill 分析当前资料
```

### 配置 MCP

MCP 默认关闭。准备好本地或远程 MCP Server 后，在 `.env` 中配置。

本地 stdio MCP：

```dotenv
AGENT_MCP__ENABLED=true
AGENT_MCP__DISCOVERY_TIMEOUT_SECONDS=10
AGENT_MCP__SERVERS={"local_demo":{"transport":"stdio","command":"python","args":["/absolute/path/to/server.py"]}}
```

远程 MCP：

```dotenv
AGENT_MCP__ENABLED=true
AGENT_MCP__SERVERS={"remote_demo":{"transport":"streamable_http","url":"https://example.com/mcp","headers":{"Authorization":"Bearer <token>"}}}
```

MCP 工具会在 Agent Server 初始化时发现一次，之后通过 `/mcp` 或 CLI 的 `/mcp` 查看结果。修改配置后需要重启 Server。

联网搜索也通过 MCP 接入。需要配置一个实际提供搜索能力的 MCP Server，并确保它暴露名称包含
`web_search`、`internet_search`、`browser_search` 或 `search_web` 的 Tool。项目不会把当前
Codex/开发环境的浏览器能力自动注入 Agent Server。

配置完成后，可按以下步骤确认：

```bash
uv run agent serve
uv run agent chat --user alice
```

```text
/mcp
/search-mode web
最新黄金价格是多少？
```

`/mcp` 应显示已发现的联网搜索 Tool；修改 MCP 配置后必须重启 Agent Server。不要把 API Key
直接提交到 Git，建议通过本地 `.env` 或部署环境的 Secret 注入。

### 测试与质量检查

```bash
uv run pytest -q
uv run pytest --cov=src --cov-report=term-missing -q
uv run ruff check .
uv run pyright
```

截至 2026-08-27，当前环境验证结果为：

- 默认测试 `183 passed, 2 deselected`；
- 真实 PostgreSQL + pgvector 集成测试 `1 passed`；
- FastEmbed 中文模型完成真实下载、索引和 `[S1]` 引用检索冒烟测试；
- 未执行项是需要真实对话模型配置的 Live Model 测试；
- Ruff 检查通过；
- Pyright 检查通过；

当前 API 面向本地可信用户，不等同于正式身份认证和生产级多租户系统。
