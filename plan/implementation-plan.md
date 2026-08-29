# 通用 Agent 项目架构与实现计划

## 1. 项目目标

本项目以 LangGraph + LangChain 为主要技术基础，抽取一套与业务无关的通用 Agent Harness，并通过业务组件快速适配不同类型的 Agent。

当前只在 `business/` 中实现一个 `knowledge_assistant`，用于调试和验证以下三类能力：

1. Agent 基本功能：Agent Loop、Tool Use、权限、上下文、记忆、任务与错误恢复。
2. RAG：知识入库、检索、上下文增强、来源引用和检索评测。
3. Agent Teams（Multi-Agent）：任务委派、上下文隔离、并行执行、结果审核和失败恢复。

`knowledge_assistant` 是本项目唯一的业务 Agent，也是通用 Harness 的集成测试项目。它负责检索资料、分析信息、制定计划并生成经过审核的报告。未来开发智能客服、旅游规划、学习陪伴、智能问诊、心理陪伴、Coding、监控等场景时，复制本模板并替换 `business/knowledge_assistant/` 的业务实现，继续复用 `harness/` 和 `services/`。每个项目只装配一个对外业务 Agent，Agent Teams 作为该 Agent 的内部协作机制。

---

## 2. 总体架构

```text
src/                                    # 源代码目录
├── harness/                            # Agent 通用核心
│   ├── __init__.py                     # 对外导出通用接口
│   ├── profile.py                      # AgentProfile 通用定义
│   ├── agent_loop.py                   # AgentRuntime 接口
│   ├── conversation.py                 # Conversation 生命周期、Run 和会话所有权
│   ├── state.py                        # 通用运行状态
│   ├── messages.py                     # 消息、ToolCall、ToolMessage
│   ├── graph.py                        # LangGraph 主图、节点和条件路由
│   ├── model.py                        # 模型调用抽象
│   ├── tool_use.py                     # 工具协议、注册和执行
│   ├── permissions.py                  # 权限协议、决策和审批管线
│   ├── context.py                      # 上下文构建
│   ├── system_prompt.py                # Prompt 组合
│   ├── hooks.py                        # Hook 事件、注册和触发
│   ├── error_recovery.py               # 错误分类
│   │
│   └── capabilities/                   # 可插拔通用能力
│       ├── todo_write.py                # TodoWrite 计划管理
│       ├── subagent.py                  # Subagent 创建和上下文隔离
│       ├── skill_loading.py             # Skill 扫描和按需加载
│       ├── context_compact.py           # 上下文压缩
│       ├── memory.py                    # 跨会话记忆
│       ├── task_system.py               # 任务创建、认领和完成
│       ├── background_tasks.py          # 后台任务提交和结果回注
│       │
│       ├── rag/                         # RAG 通用能力
│       │   ├── contracts.py             # Retriever、Document 等接口
│       │   ├── pipeline.py              # 通用检索流程
│       │   └── citations.py             # 来源引用模型
│       │
│       └── agent_teams/                 # Agent Teams 通用能力
│           ├── contracts.py             # Agent、Task、Result 协议
│           ├── team.py                  # Lead、Teammate 注册和任务委派
│           ├── message_bus.py            # 消息总线
│           ├── team_protocols.py         # Agent 间消息协议
│           └── autonomous_agents.py      # 自治 Agent
│
├── business/                           # 具体业务 Agent
│   └── knowledge_assistant/            # 当前唯一业务，用于调试
│       ├── __init__.py                 # 导出知识助手 Profile
│       ├── profile.py                  # Agent 装配配置
│       ├── state.py                    # 知识任务业务状态
│       ├── schemas.py                  # 文档、任务、报告结构
│       ├── system_prompt.py             # 知识助手 Prompt 片段
│       ├── context.py                  # 知识库和任务上下文
│       ├── permission_rules.py         # 具体业务权限规则
│       │
│       ├── tools/                       # 知识助手业务工具
│       │   ├── calculator.py           # 数学计算
│       │   ├── document_search.py      # 搜索知识文档
│       │   ├── file_reader.py          # 读取受限文件
│       │   └── report_writer.py        # 生成和保存报告
│       │
│       ├── agent_teams/                 # 知识助手团队角色
│       │   ├── lead.py                 # 主 Agent / Supervisor
│       │   ├── researcher.py           # 资料检索
│       │   ├── analyst.py              # 比较、计算和分析
│       │   └── reviewer.py             # 检查结论与引用
│       │
│       └── knowledge/                   # 本地调试知识库
│           ├── documents/              # 调试知识文档
│           └── fixtures/               # 测试数据
│
├── services/                           # 具体技术和外部服务实现
│   ├── config.py                       # 配置加载
│   ├── models.py                       # Anthropic 等模型适配
│   ├── model_gateway.py                # 模型并发、重试和降级路由
│   ├── checkpoint.py                   # LangGraph Checkpointer
│   ├── stores.py                       # 会话、记忆和任务存储
│   ├── artifacts.py                    # 报告等文件产物存储
│   ├── background_tasks.py             # 后台任务执行实现
│   ├── cron_scheduler.py               # 定时调度
│   ├── message_bus.py                  # Redis 消息总线实现
│   ├── observability.py                # 日志、指标和链路追踪
│   ├── security.py                     # 身份、角色、密钥和审计实现
│   ├── mcp_tools.py                    # MCP 连接适配
│   │
│   └── rag/                             # RAG 技术实现
│       ├── embeddings.py               # Embedding 模型适配
│       ├── vector_store.py             # pgvector 适配
│       ├── ingestion.py                # 文档入库
│       ├── splitter.py                 # 文档切分
│       └── reranker.py                 # 可选重排
│
└── entrypoints/                        # 运行入口和应用装配
    ├── bootstrap.py                    # 创建 Runtime 和共享服务
    ├── cli.py                          # 多用户 CLI 客户端和本地调试入口
    ├── api.py                          # 单进程 Agent Server；后续扩展生产 API
    ├── worker.py                       # 生产化阶段的后台任务 Worker
    └── indexer.py                      # RAG 数据导入入口
```

### 2.1 各层职责

| 目录 | 职责 | 不应该包含的内容 |
| --- | --- | --- |
| `harness/` | 定义 Agent Loop 以及可叠加的 Harness 能力 | 知识助手、客服等具体业务规则；具体数据库实现 |
| `business/` | 定义 Agent 做什么、使用哪些工具、遵守哪些业务规则 | LangGraph 底层持久化；通用模型客户端 |
| `services/` | 实现模型、数据库、向量库、Background Tasks 和外部连接 | Agent 的业务 System Prompt 和装配逻辑 |
| `entrypoints/` | 装配依赖并通过 CLI、API、Worker 启动系统 | 核心业务判断和复杂流程 |

核心边界：

```text
Agent Loop  决定：模型、工具和消息如何持续循环
Harness     决定：Agent 具备哪些可叠加能力
Business    决定：Agent 在什么业务中工作
Tool Use    决定：Agent 能对外部世界做什么
Permission  决定：Agent 被允许做什么
Context     决定：Agent 当前知道什么
Services    决定：上述能力由哪些具体技术实现
```

### 2.2 命名审查结果

下列名称与 `learn-claude-code` 中的概念含义相同，因此统一使用该项目的章节术语：

| 原命名 | 调整后 | 对应章节或概念 |
| --- | --- | --- |
| `agent_core/` | `harness/` | Harness Engineering；模型之外让 Agent 感知和行动的运行设施 |
| `runtime.py` | `agent_loop.py` | s01 Agent Loop |
| `tools.py` | `tool_use.py` | s02 Tool Use |
| `permissions.py` | 保留 | s03 Permission |
| `events.py`、`middleware.py` | `hooks.py` | s04 Hooks；事件定义和注册触发放在同一模块 |
| `todo.py` | `todo_write.py` | s05 TodoWrite |
| `skills.py` | `skill_loading.py` | s07 Skill Loading |
| `compact.py` | `context_compact.py` | s08 Context Compact |
| `prompt.py`、`prompts.py` | `system_prompt.py` | s10 System Prompt |
| `errors.py` | `error_recovery.py` | s11 Error Recovery |
| `tasks.py` | `task_system.py` | s12 Task System |
| `background.py`、`jobs.py` | `background_tasks.py` | s13 Background Tasks |
| `scheduler.py` | `cron_scheduler.py` | s14 Cron Scheduler |
| `multi_agent/` | `agent_teams/` | s15 Agent Teams |
| `supervisor.py` | `agent_teams/team.py` 中的 Lead | s15 中的 Lead Agent |
| `protocols.py` | `team_protocols.py` | s16 Team Protocols |
| `event_bus.py` | `message_bus.py` | s15 中的 MessageBus |
| `mcp.py` | `mcp_tools.py` | s19 MCP Tools |

以下名称没有 `learn-claude-code` 中的同义概念，因此保留其原有技术名称：

- `graph.py`：LangGraph 的图、节点和条件路由。
- `checkpoint.py`：LangGraph 的状态持久化机制。
- `rag/`：本项目新增的检索增强能力。
- `profile.py`：本模板用于业务装配的配置抽象。
- `entrypoints/`：CLI、API、Worker 等应用入口。
- Comprehensive Agent Turn 作为最终集成验收概念保留，不单独设置 `agent_turn.py`；业务装配由 `profile.py` 完成，执行由 `harness/agent_loop.py` 完成。

### 2.3 架构精简结果

审查后删除或合并以下冗余文件：

| 删除项 | 处理方式 | 原因 |
| --- | --- | --- |
| `harness/hook_events.py` | 合并到 `harness/hooks.py` | Hook 事件类型和 Hook 注册、触发属于同一能力，当前规模无需拆分 |
| `harness/routing.py` | 合并到 `harness/graph.py` | 条件路由只服务于 LangGraph 主图，单独文件会增加跳转成本 |
| `business/base.py` | `AgentProfile` 移到 `harness/profile.py` | AgentProfile 是所有业务共享的 Harness 契约，不属于业务层 |
| `agent_teams/registry.py`、`lead.py`、`teammates.py` | 合并为 `agent_teams/team.py` | 三者共同负责团队成员注册、Lead 调度和 Teammate 委派，首版无需过度拆分 |
| `business/knowledge_assistant/tools/tasks.py` | 直接复用 `harness/capabilities/task_system.py` | 目前没有知识助手特有的任务规则，避免重复包装通用 Task System |

保留的相似文件具有不同层级职责：

- `harness/model.py` 定义模型调用协议，`services/models.py` 实现 Anthropic 等具体适配。
- `services/model_gateway.py` 包装 `ModelProvider`，统一处理本机跨进程并发、临时错误重试和备用模型降级；Agent Loop 不感知具体 Provider。
- `harness/capabilities/background_tasks.py` 负责 Agent Loop 中的提交和结果回注，`services/background_tasks.py` 负责实际后台执行。
- `harness/capabilities/agent_teams/message_bus.py` 定义团队消息接口和进程内实现，`services/message_bus.py` 提供 Redis 等生产实现。
- `harness/capabilities/rag/` 定义检索流程和引用协议，`services/rag/` 实现 Embedding、向量库及文档入库。

---

## 3. Knowledge Assistant 调试场景

知识工作助手需要覆盖一个完整的资料研究与报告生成流程：

```text
导入业务资料
    ↓
用户提出研究或分析任务
    ↓
Lead Agent 拆分任务
    ↓
Researcher 检索并整理资料
    ↓
Analyst 比较、计算和形成结论
    ↓
Reviewer 检查结论与引用
    ↓
Lead Agent 汇总最终报告
    ↓
经权限确认后保存报告
    ↓
可选生成后续跟踪任务
```

### 3.1 调试工具

| 工具 | 来源 | 启用阶段 | 用途 | 默认权限 |
| --- | --- | --- | --- | --- |
| `calculator(expression)` | `business/.../tools/calculator.py` | 第一阶段 | 执行受限数学计算 | 自动允许 |
| `file_reader(path)` | `business/.../tools/file_reader.py` | 第一阶段 | 读取授权范围内的本地文件 | 按路径策略允许 |
| `report_writer(title, content)` | `business/.../tools/report_writer.py` | 第一阶段 | 生成并保存分析报告 | 需要确认或按策略允许 |
| `create_task(subject, description)` | `harness/capabilities/task_system.py` | 第一阶段 | 创建研究任务 | 按业务策略允许 |
| `list_tasks()` / `claim_task(task_id)` | `harness/capabilities/task_system.py` | 第一阶段 | 查看或认领任务 | 按业务策略允许 |
| `complete_task(task_id)` | `harness/capabilities/task_system.py` | 第一阶段 | 完成任务并保存状态 | 按业务策略允许 |
| `document_search(query, filters)` | `business/.../tools/document_search.py` | 第二阶段 | 调用 RAG Pipeline 检索知识库 | 自动允许 |

第一阶段先验证计算、受限文件访问、写入产物、Task System、结构化输出、Error Recovery 和 Permission；第二阶段接入 `document_search` 后再验证 RAG。

---

## 4. 实现计划

项目按四个正式阶段推进。RAG 和 Agent Teams 仍分别位于第二、第三阶段；此前列为
“本期不做”的生产能力统一进入第四阶段，避免在单 Agent 主链路尚未稳定时提前引入
分布式队列、后台 Worker 和复杂身份系统。

| 阶段 | 建设重点 | 阶段结果 |
| --- | --- | --- |
| 第一阶段 | 工程骨架和基本 Agent 功能 | 可独立运行的单 Agent MVP |
| 第二阶段 | RAG | 能够基于通用文档知识库检索并生成带引用的回答 |
| 第三阶段 | Agent Teams | Lead Agent 能够委派、并行执行并审核 Teammate 任务 |
| 第四阶段 | 生产化与高级交互 | 多 Worker、持久任务、正式身份、实时事件和故障恢复 |

综合 Agent Turn 不单独占用开发阶段，而是在第四阶段完成后作为全系统验收。

各阶段对应总体架构的建设范围：

| 阶段 | 主要目录 |
| --- | --- |
| 第一阶段 | `harness/` 基础模块、Conversation、TodoWrite/Skill Loading/Context Compact/Memory/Task System、Knowledge Assistant 单 Agent、通用 Services、Bootstrap、单进程 API 和多用户 CLI |
| 第二阶段 | `harness/capabilities/rag/`、`services/rag/`、`knowledge/`、`document_search.py`、`entrypoints/indexer.py` |
| 第三阶段 | `subagent.py`、`agent_teams/`、业务团队角色、进程内 MessageBus |
| 第四阶段 | PostgreSQL Checkpoint/Store、正式 API、SSE/WebSocket、分布式队列、Worker、Background Tasks、Cron、生产 MessageBus、身份与综合验收 |

## 4.1 第一阶段：工程骨架和基本 Agent 功能

详细任务、验证场景和退出条件见 [第一阶段实施计划](./phase-1-implementation-plan.md)。

阶段状态：**已完成（2026-08-27）**。M1～M5 的自动化、CLI 手工验收和第一阶段退出条件
均已通过。

### 4.1.1 项目骨架和核心契约

#### 实现内容

- 创建 Python 工程、依赖分组和配置体系。
- 在 `harness/profile.py` 定义 `AgentProfile`，描述模型、System Prompt、Tools、PermissionRule、ContextProvider 和 Capability。
- 在 `state.py`、`messages.py`、`hooks.py` 定义 `AgentState`、`Message`、`ToolUse`、`ToolResult` 和 `HookEvent`。
- 定义 `AgentLoop`、`ModelProvider`、`Tool`、`PermissionRule`、`Store` 等协议。
- 在 `business/knowledge_assistant/profile.py` 装配唯一业务 Agent 的能力。
- 在 `entrypoints/bootstrap.py` 创建 Agent Loop、模型、Checkpoint、Store 和其他共享服务。
- 在 `harness/conversation.py` 管理 `user_id`、`conversation_id/thread_id` 和 `run_id`。
- 通过单进程 FastAPI Agent Server 支持多个可信 CLI 客户端同时使用。
- 建立单元测试、类型检查和代码质量检查。

#### 验收标准

- 可以通过 CLI 加载 `knowledge_assistant`。
- 可以完成一次不使用工具的模型问答。
- 不同 CLI 用户的 Conversation、Checkpoint 和 Artifact 目录相互隔离。
- 不同 Conversation 可以异步执行，同一 Conversation 同时只允许一个主 Run。
- `business/` 不直接依赖具体模型厂商或数据库客户端。
- `harness/` 不包含任何知识助手业务概念。
- `AgentProfile` 只描述装配关系，不包含 Agent Loop 实现。

---

### 4.1.2 Agent Loop

使用 LangGraph 构建基础执行图：

```text
START
  ↓
加载 Checkpoint
  ↓
构建 System Prompt 和 Context
  ↓
调用模型
  ↓
是否包含 Tool Use？
  ├── 否 → 保存状态 → 输出结果 → END
  └── 是
        ↓
      权限检查
        ├── 拒绝 → 生成拒绝结果 ─────────┐
        ├── 待确认 → interrupt()          │
        └── 允许 → 执行工具               │
                         ↓                │
                      ToolResult ─────────┘
                         ↓
                      再次调用模型
```

实现：

- 多轮消息和请求完成后的事件集合；Token 级流式输出延后到第四阶段。
- 单个、多个以及可并行工具调用。
- Tool Use ID 与 ToolResult 严格配对。
- 最大循环次数、Token 预算和请求级超时；远程取消延后到第四阶段。
- 模型及工具的结构化输入输出。
- 每一轮执行状态可持久化和恢复。
- System Prompt 按基础规则、Tool、业务、Skill、Memory、Runtime Context 的固定顺序组装。
- Runtime Context 按优先级去重和预算选择，业务层只暴露授权资料的相对名称。

### 4.1.3 Conversation Loop 与多用户 CLI

第一阶段在 Agent Loop 外增加统一 Conversation 层：

```text
CLI Clients
    ↓ HTTP
单进程 FastAPI Agent Server
    ↓
ConversationService
    ↓
每用户复用的 AgentLoop + SQLite Checkpointer
    ↓
共享 ModelGateway
```

- `conversation_id` 直接作为 LangGraph `thread_id`。
- 一次用户输入到最终回答使用一个应用级 `run_id`；权限恢复沿用原 `run_id`。
- 多个可信 CLI 用户通过 `--user` 连接同一个 Agent Server。
- 每个用户首次访问时创建 User Runtime，此后复用该用户的 AgentLoop；ModelGateway 全局共享。
- 用户 Artifact 写入 `artifacts/<user_id>/`；Conversation 所有权、状态和待审批信息由共享的
  本地 SQLite 索引校验，每个用户的完整 AgentState 写入各自的 Checkpoint 数据库。
- 不同 Conversation 可以通过异步 API 并发；同一 Conversation 已有活跃 Run 时返回冲突，不排队。
- CLI 提供 `/new`、`/list`、`/switch`、`/delete`、`/exit`；权限请求通过单独的 HTTP Resume 请求继续同一 Run；
  Server 重启后可通过 `/switch` 回到 `waiting_permission` 会话继续审批。
- 删除 Conversation 时同步清理 Registry 和 Checkpoint；活跃 Run 未结束时拒绝删除。
- 第一阶段只返回请求级事件集合，不实现 Token Streaming、SSE 或 WebSocket。

该入口只面向本地可信调试用户，不等价于正式账号和多租户安全系统。

### 4.1.4 权限机制

权限系统分成操作风险类别和权限决策结果两个维度。

操作风险类别：

```text
READ       只读操作，默认自动允许
WRITE      修改业务数据，由业务 PermissionRule 判断
EXTERNAL   对外发送或调用产生实际影响，需要确认
DANGEROUS  高风险操作，默认禁止
```

权限决策结果：

```text
ALLOW          直接允许执行
DENY           直接拒绝执行
ASK            暂停并请求用户审批
PASSTHROUGH    当前规则不作决定，继续交给后续规则
```

三个目录的职责保持独立：

| 位置 | 职责 |
| --- | --- |
| `harness/permissions.py` | 定义 `PermissionBehavior`、`PermissionResult`、`PermissionRule` 及通用权限检查和审批管线 |
| `business/knowledge_assistant/permission_rules.py` | 实现文档访问、文件读取、报告写入和外部发布等业务权限规则 |
| `services/security.py` | 提供用户身份、角色、密钥、授权数据查询和审计存储等技术实现 |

执行顺序为：工具调用进入通用 Permission Pipeline，依次执行通用规则和业务 `PermissionRule`；结果为 `ASK` 时使用 LangGraph `interrupt()` 暂停，通过同一 `thread_id` 接收审批结果并恢复执行。

### 4.1.5 Hooks

第一阶段先实现与 `learn-claude-code` 一致的 Hook 事件：

- `UserPromptSubmit`
- `PreToolUse`
- `PostToolUse`
- `Stop`

Hook 事件类型、注册和触发统一放在 `harness/hooks.py`。Hooks 用于实现日志、Permission、结果检查和上下文注入，业务 Agent 不直接修改 Agent Loop。Error Recovery 由 `error_recovery.py` 处理；底层可以使用 LangChain Middleware 适配这些事件，但业务层统一使用 Hooks 术语。

### 4.1.6 单 Agent 扩展能力

- TodoWrite 与任务规划。
- Skill Loading。
- 短期会话记忆。
- 长期用户偏好和任务记录。
- Context Compact。
- Checkpoint 和断点恢复。
- Error Recovery：模型重试、工具重试和模型降级。
- Model Gateway：主模型达到本地并发上限时直接选择独立额度的备用模型；HTTP 429 重试一次后降级，超时、连接错误和 5xx 最多重试两次，并向入口发送可见事件；所有模型失败时向 Chat 展示重试过程和最终原因。
- MCP Tools 接入。

#### 第一阶段验收标准

- 知识助手可以调用计算、文件读取、报告写入和 Task System 工具；RAG 检索留到第二阶段。
- 权限不足时可以暂停、拒绝或等待用户确认。
- 模型或工具发生临时错误后可以按照策略重试。
- 进程中断后可以从 Checkpoint 恢复。
- 超长对话可以压缩，同时保留任务目标、资料来源和关键结论。
- TodoWrite 能维护当前执行计划，Task System 能持久化任务依赖、认领和完成状态。
- Skill 只在需要时加载，正文不长期占用 System Prompt。
- MCP Tools 能完成发现、组装和调用，并经过同一 Permission Pipeline。
- 多个 CLI 客户端通过单进程 API 使用统一 Bootstrap；不同用户状态隔离，共享模型网关。

---

## 4.2 第二阶段：RAG

当天可落地的任务拆解、验证方案和延期边界见
[第二阶段实施计划](./phase-2-implementation-plan.md)。

阶段状态：**已完成（2026-08-27）**。M6 默认自动化、真实 pgvector 集成、FastEmbed
索引检索、用户隔离和引用验收均已通过。

### 4.2.1 能力边界

RAG 按总体架构分成业务知识、通用协议和技术实现三部分：

```text
business/knowledge_assistant/knowledge/
    提供业务文档和元数据
                 ↓
services/rag/
    完成加载、切分、向量化和实际存储

用户查询
   ↓
business/.../tools/document_search.py
   ↓
harness/capabilities/rag/
    执行统一检索流程、Token 控制和引用生成
   ↓
services/rag/vector_store.py
```

具体职责：

- `contracts.py` 定义 Retriever、Document 和检索结果协议。
- `pipeline.py` 编排查询、过滤、召回、重排和上下文生成。
- `citations.py` 生成并校验来源引用。
- `document_search.py` 是知识助手的业务 Tool，只调用通用 RAG Pipeline，不直接访问 pgvector。
- `services/rag/` 负责 Embedding、向量库、切分、入库和可选 Rerank。
- `business/knowledge_assistant/profile.py` 在第二阶段注册 `document_search` 和 RAG Capability。

### 4.2.2 文档入库

实现：

- `entrypoints/indexer.py` 接收导入命令并调用 `services/rag/ingestion.py`。
- 第一版支持 Markdown、TXT；PDF 作为后续增强。
- 根据标题、段落和 Token 数量进行语义切分。
- 为 Chunk 保存以下元数据：
  - `document_id`
  - `knowledge_base_id`
  - `section`
  - `category`
  - `tags`
  - `source`
  - `version`
  - `tenant_id` 或 `user_id`
- 批量生成 Embedding。
- 支持增量更新、版本替换、删除和重建索引。

第二阶段只交付可信管理员手工触发的同步 Indexer。生产环境中的用户上传 API、对象存储事件、
CDC/Webhook、事务性 Index Job/Outbox、持久队列、独立 Index Worker、版本原子切换、删除事件、
定时对账和索引状态查询延后到第四阶段 M10；Agent Server 只负责检索，不直接监听目录或执行批量
Embedding。详细设计见[第二阶段实施计划 5.5](./phase-2-implementation-plan.md#55-后续生产化文档更新链路本阶段不实现)。

建议入口：

```bash
agent index knowledge-assistant ./src/business/knowledge_assistant/knowledge/documents
```

### 4.2.3 查询流程

```text
用户问题
  ↓
查询规范化或改写
  ↓
租户、知识库、分类、标签等元数据过滤
  ↓
向量召回 Top-K
  ↓
可选关键词混合检索
  ↓
可选 Rerank
  ↓
去重并控制 Token 预算
  ↓
注入模型上下文
  ↓
生成带来源引用的回答
```

### 4.2.4 安全与可靠性

- 检索文档被视为不可信数据，不允许覆盖 System Prompt。
- 用户私有文档必须通过 `tenant_id` 或 `user_id` 强制过滤。
- 没有检索结果时明确降级，不伪造知识来源。
- 保存检索查询、命中文档、相关度分数、耗时和最终引用。
- Embedding 模型和向量维度必须版本化。

### 4.2.5 测试与验收

- 建立固定问题、期望文档和期望答案的数据集。
- 测试 Recall@K、MRR、引用正确率和无依据回答率。
- 文档更新后，旧版本不得继续参与检索。
- 用户 A 的私有资料不能被用户 B 检索。
- 回答中的引用必须能够定位到实际文档及 Chunk。
- 关闭 RAG Capability 时，Knowledge Assistant 仍能以第一阶段单 Agent 模式运行。

---

## 4.3 第三阶段：Agent Teams

详细任务、验证方案和退出条件见[第三阶段实施计划](./phase-3-implementation-plan.md)。

阶段状态：**计划中**。当前仅保留 Agent Teams 的目录和模块骨架，尚未接入 Agent Loop、Profile 和运行时。

第一版采用 Lead Agent + Subagents/Teammates 模式，不直接构建完全自治的对等 Agent 网络。

```text
                        ┌→ Researcher ───────┐
用户 → Lead Agent ──────┼→ Analyst ──────────┼→ Reviewer → Lead Agent → 用户
                        └→ Knowledge Tools ──┘
```

### 4.3.1 Agent 职责

#### Lead Agent

- 作为主 Agent 和团队 Lead。
- 理解用户目标并拆分研究或分析任务。
- 决定调用哪个 Subagent 或 Teammate。
- 汇总 Subagent/Teammate 结果并与用户交互。
- 控制任务预算和最大委派深度。

#### Researcher

- 执行知识检索和资料整理。
- 返回结构化知识摘要和来源。
- 默认不修改用户数据。

#### Analyst

- 对检索到的资料进行比较、计算和归纳。
- 返回结构化分析、关键结论和结论依据。

#### Reviewer

- 检查内容正确性。
- 检查来源引用是否真实。
- 检查分析结论是否由资料和计算支撑。
- 不合格时返回明确的修改意见。

### 4.3.2 通用能力

在 `harness/capabilities/agent_teams/` 中实现：

- `contracts.py` 定义 Team、Teammate、Task、Agent Result 等协议。
- `team.py` 负责 Teammate 注册、Lead 调度和任务委派。
- `message_bus.py` 定义消息接口并提供进程内实现。
- `team_protocols.py` 实现请求—响应、计划审批、关闭和 Permission 冒泡协议。
- Subagent/Teammate 独立状态与上下文隔离。
- Lead Agent 路由和结果聚合。
- 多个无依赖任务的并行执行。
- 最大委派深度、调用次数、耗时和 Token 预算。
- Subagent/Teammate 超时、重试、取消和失败降级。
- Agent Team 调用关系和成本的完整追踪。

业务和服务层负责：

- `business/knowledge_assistant/agent_teams/` 定义 Lead、Researcher、Analyst 和 Reviewer 的业务 Prompt、Tools 与输出 Schema。
- `knowledge_assistant/profile.py` 注册团队角色并启用 Agent Teams Capability。
- 第三阶段只使用进程内 MessageBus；`services/message_bus.py` 的 Redis 持久化实现延后到第四阶段。

### 4.3.3 上下文隔离原则

Lead Agent 不应把完整会话无条件传给 Subagent/Teammate。每次委派只传递：

- 任务目标。
- 必要的业务参数。
- 经过筛选的上下文。
- 输出 Schema。
- 可用工具和权限。
- 时间、Token 和递归预算。

Subagent/Teammate 只返回结构化结果和必要产物，不直接修改 Lead Agent 的全部状态。

### 4.3.4 同步到异步的演进

第一版：

- 使用 LangGraph Subgraph 表达每个 Subagent。
- 使用 Agent-as-Tool 方式由 Lead Agent 调用 Subagent。
- 使用 `team.py`、MessageBus 和 Team Protocols 处理需要持续通信的 Teammate。
- 使用 `asyncio.TaskGroup` 并行执行无依赖任务。

生产化版本（第四阶段）：

- 长任务提交到后台 Worker。
- Redis/MessageBus 通知任务状态。
- `entrypoints/worker.py` 执行长任务，`entrypoints/api.py` 提供任务状态和流式结果接口。
- 支持任务查询、取消和恢复。
- 根据业务需要增加 Agent Handoff 和人工接管。

### 4.3.5 测试与验收

- Lead Agent 可以根据意图选择正确的 Subagent/Teammate。
- 多个 Researcher 任务以及无依赖的 Researcher、Analyst 任务可以并行执行。
- Subagent/Teammate 无法访问与任务无关的主会话和工具。
- 任意 Subagent/Teammate 失败不会破坏 Lead Agent 状态。
- Reviewer 可以退回不合格结果并触发有限次数重做。
- Team Protocols 能正确处理请求—响应、计划审批和关闭流程。
- Teammate 的 `ASK` 权限请求能够冒泡到 Lead Agent，并将审批结果返回原任务。
- 正常的 Permission interrupt/resume 不会重复执行已完成节点；崩溃边界下的副作用幂等在第四阶段保证。
- 每次委派均可查看输入、输出、延迟、Token 和错误。

---

## 4.4 第四阶段：生产化与高级交互

第四阶段按依赖关系分为三个优先级，不要求一次性全部启用。

### P0：持久化、多用户安全与实时入口

- PostgreSQL Checkpointer 和 Conversation/Run Store。
- 正式身份认证、Conversation 所有权、租户数据与私有知识隔离。
- SSE 运行事件；确有双向实时控制需求时再增加 WebSocket。
- `conversation_id/thread_id/run_id` 的持久记录、日志关联和 OpenTelemetry Trace。
- Token 级流式输出、远程取消和服务重启恢复。

### P1：可靠异步任务与水平扩展

- 在引入自动重新投递前补充 `message_id`、副作用工具幂等键和执行状态。
- 多 FastAPI Worker、跨进程同 Thread 串行控制和模型额度协调。
- PostgreSQL 任务表或 Redis Streams、Agent Worker、Background Tasks 和 Cron Scheduler。
- 受预算、租约和停止条件约束的 Autonomous Agents。
- 任务 lease、heartbeat、超时、取消、重试、重新投递和恢复。
- Redis/持久化 MessageBus 仅在进程内 MessageBus 无法满足部署需求时启用。

### P2：高级 Conversation 交互

- `/btw` 只读旁路 Run，不写回主 Thread。
- Conversation 分支、重新生成、Steering 和 Rollback。
- 多客户端同时操作同一 Conversation 时的冲突策略。
- Conversation 标题、归档、跨设备恢复和前端界面。

`s18 Worktree Isolation` 继续作为未来 Coding Agent 的业务专属增强，不并入 Knowledge
Assistant 的生产化阶段。

## 4.5 最终综合验收：Comprehensive Agent Turn

最终使用以下流程验证三类能力能够协同运行：

1. 用户导入业务或项目资料。
2. Indexer 建立知识索引。
3. 用户提出研究、比较或分析任务。
4. Lead Agent 拆分任务并调用 Researcher。
5. Researcher 使用 RAG 查找资料并返回引用。
6. Analyst 比较信息、执行必要计算并生成初步结论。
7. Reviewer 检查结论、计算过程和来源引用。
8. 不合格结果在预算范围内退回修改。
9. Lead Agent 汇总经过审核的最终报告。
10. 系统在权限确认后保存报告及任务记录。
11. 生产化部署启用时，后台任务根据需要生成后续跟踪任务。

综合验收应同时覆盖流式输出、Permission、RAG、Agent Teams、持久化、Error Recovery、Background Tasks 和可观测性。

---

## 5. 功能与原项目章节对应关系

| 原功能 | 当前架构位置 | 实现阶段 |
| --- | --- | --- |
| s01 Agent Loop | `harness/agent_loop.py`、`harness/graph.py` | 基本功能 |
| s02 Tool Use | `harness/tool_use.py` | 基本功能 |
| s03 Permission | `harness/permissions.py` + 业务 `permission_rules.py` | 基本功能 |
| s04 Hooks | `harness/hooks.py` | 基本功能 |
| s05 TodoWrite | `harness/capabilities/todo_write.py` | 基本功能 |
| s06 Subagent | `harness/capabilities/subagent.py` | Agent Teams |
| s07 Skill Loading | `harness/capabilities/skill_loading.py` | 基本功能 |
| s08 Context Compact | `harness/capabilities/context_compact.py` | 基本功能 |
| s09 Memory | `harness/capabilities/memory.py` | 基本功能 |
| s10 System Prompt | `harness/system_prompt.py` | 基本功能 |
| s11 Error Recovery | `harness/error_recovery.py`、Hooks | 基本功能 |
| s12 Task System | `harness/capabilities/task_system.py` | 基本功能 |
| s13 Background Tasks | `harness/capabilities/background_tasks.py`、`services/background_tasks.py` | 生产化阶段 |
| s14 Cron Scheduler | `services/cron_scheduler.py` | 生产化阶段 |
| s15 Agent Teams | `harness/capabilities/agent_teams/team.py`、`harness/capabilities/agent_teams/message_bus.py` | Agent Teams |
| s16 Team Protocols | `harness/capabilities/agent_teams/team_protocols.py` | Agent Teams |
| s17 Autonomous Agents | `harness/capabilities/agent_teams/autonomous_agents.py`、Worker、Budget Limits | 生产化阶段 |
| s18 Worktree Isolation | 未来 `business/coding/` | 当前不实现 |
| s19 MCP Tools | `services/mcp_tools.py` | 基本功能增强 |
| s20 Comprehensive Agent Turn | Knowledge Assistant 综合流程 | 最终验收 |
| RAG | `harness/capabilities/rag/` + `services/rag/` | RAG 阶段 |

`s18 Worktree Isolation` 是 Coding Agent 的专属能力，与知识助手无关。当前仅预留任务隔离和 Artifact 接口，未来增加 `business/coding/` 时再实现 Git Worktree 适配。

---

## 6. 技术选型

| 范围 | 推荐技术 | 主要用途 | 采用策略 |
| --- | --- | --- | --- |
| 语言 | Python 3.12 | 主开发语言 | 固定主版本 |
| 项目管理 | uv + `pyproject.toml` | 依赖、虚拟环境、锁文件 | 默认采用 |
| Agent 编排 | LangGraph `StateGraph` | Agent Loop、路由、状态恢复 | 核心依赖 |
| Agent 组件 | LangChain | Model、Tool、Middleware 桥接和 RAG 适配 | 核心依赖 |
| 模型 | Claude + `langchain-anthropic` | Tool Use、结构化输出、流式响应 | 第一模型适配 |
| 本地模型网关 | `ModelProvider` 包装器 + 系统文件锁 | 跨进程并发、指数退避重试、备用模型降级 | 个人项目采用 |
| 数据模型 | Pydantic v2 | AgentProfile、State、工具参数和 Agent 协议 | 默认采用 |
| 配置 | pydantic-settings | 环境变量和配置校验 | 默认采用 |
| Skill 配置 | PyYAML | Skill frontmatter 和配置解析 | 第一阶段采用 |
| 本地 Checkpoint | `langgraph-checkpoint-sqlite` | 单机开发和调试 | 开发环境 |
| 生产 Checkpoint | `langgraph-checkpoint-postgres` | 持久化和并发 | 生产环境 |
| ORM/迁移 | SQLAlchemy 2 + Alembic | 业务数据访问和迁移 | 数据库启用时采用 |
| PostgreSQL 驱动 | psycopg 3 | PostgreSQL 异步访问 | 默认采用 |
| 文件产物 | 本地文件系统，S3 兼容存储可选 | 报告、任务输出和大工具结果 | 本地优先 |
| 向量数据库 | PostgreSQL + pgvector | RAG 向量检索 | 首选方案 |
| LangChain 向量适配 | `langchain-postgres` | 连接 pgvector | RAG 阶段采用 |
| Embedding | 可配置 Provider | 生成文档与查询向量 | 不绑定单一厂商 |
| 文本切分 | `langchain-text-splitters` | RAG Chunk 切分 | RAG 阶段采用 |
| CLI | Typer + Rich | 本地调试和管理命令 | 首个入口 |
| HTTP API | FastAPI + Uvicorn | 第一阶段的单进程 Agent Server；后续扩展生产接口 | 第一阶段采用 |
| 流式输出 | SSE 优先，WebSocket 可选 | 模型输出流 | 生产化阶段采用 |
| 本地并发 | asyncio + `TaskGroup` | 并行工具与子 Agent | 默认采用 |
| 后台任务 | PostgreSQL 任务表或 Redis Streams；框架按需选择 | 长任务和异步 Agent | 生产化阶段按需启用 |
| 定时任务 | APScheduler | 跟踪任务和周期调度 | 生产化阶段按需启用 |
| Agent Teams 消息 | 进程内 MessageBus / Redis | 本地通信与生产持久通信 | 第三阶段采用 |
| MCP | `langchain-mcp-adapters` | 接入 MCP Server | 基本功能增强 |
| HTTP 客户端 | HTTPX | 外部接口调用 | 默认采用 |
| 结构化日志 | structlog | 日志和上下文绑定 | 默认采用 |
| 链路追踪 | OpenTelemetry | 跨服务 Trace | 生产化阶段 |
| Agent 调试 | LangSmith | Graph、Tool、RAG 调试 | 可选 |
| 测试 | pytest + pytest-asyncio + HTTPX ASGITransport | 单元、异步和进程内 HTTP 测试 | 默认采用 |
| 代码检查 | Ruff + Pyright | Lint、格式和类型检查 | 默认采用 |
| 本地部署 | Docker Compose | PostgreSQL、pgvector、Redis | 集成测试使用 |

### 6.1 总体架构与技术选型映射

总体架构按“入口、编排、工具、数据、协作、运维”六类能力选型：

| 能力 | 技术选型 | 说明 |
| --- | --- | --- |
| 入口与服务 | Typer、Rich、HTTPX、FastAPI、Uvicorn | CLI Client/Indexer 通过 HTTP 使用 Agent Server；当前单进程运行 |
| Agent 编排 | LangGraph `StateGraph`、LangChain | 负责 Agent Loop、Tool 路由、状态恢复和模型适配 |
| 模型与安全 | Model Gateway、Pydantic、Permission Pipeline、Hooks | 统一模型重试/降级、工具校验、审批和审计 |
| RAG 数据链路 | FastEmbed、`langchain-text-splitters`、PostgreSQL + pgvector | 完成文档入库、向量检索、权限过滤和 Citation |
| 外部工具 | `langchain-mcp-adapters` | 统一接入 MCP；联网搜索需要额外配置 Web Search MCP Tool |
| Agent Teams | LangGraph Subgraph、`asyncio.TaskGroup`、进程内 MessageBus | 第三阶段实现 Subagent、并行任务和审核；生产化再引入持久总线 |
| 会话与产物 | SQLite Checkpoint/Conversation Index、本地文件系统 | 当前用于会话恢复、用户隔离和报告保存 |
| 生产化存储与任务 | PostgreSQL、Redis、Worker、S3 兼容存储 | 第四阶段用于多 Worker、后台任务、索引更新和大文件产物 |
| 可观测性与部署 | structlog、OpenTelemetry、Docker Compose | 当前结构化日志和本地部署；生产环境增加 Trace、Metrics 和告警 |

需要保持三条边界：

- `CLI Client` 是在线对话入口，`CLI Indexer` 是离线知识入库入口；两者共享项目配置，但不共享执行链路。
- MCP 是外部工具协议，不等于联网搜索服务。只有配置了实际的 Web Search MCP Tool，`web` 和 `hybrid` 模式才具备联网能力。
- PostgreSQL/pgvector 负责 RAG 数据，Conversation、Checkpoint、Task 和生产索引状态按阶段使用独立 Store 或逻辑隔离，避免把向量检索职责扩散到业务层。

### 6.2 技术选择原则

- 业务代码依赖通用协议，不直接依赖 LangGraph 节点对象。
- 模型、Embedding、向量库和 Checkpointer 均通过接口注入。
- 第一阶段多用户 CLI 使用单进程 FastAPI、进程内 Registry 和每用户 Runtime，降低启动成本。
- 本地恢复优先使用 SQLite；正式多 Worker 使用 PostgreSQL。
- 只有在需要并发、持久任务和生产部署时才引入 PostgreSQL、Redis 和 Worker。
- 不在第一版同时引入多个向量数据库或多个任务队列。
- LangSmith 属于可选调试平台，核心日志和追踪不能只依赖 LangSmith。

---

## 7. 建议依赖分组

```text
core:
  langgraph
  langchain
  langchain-anthropic
  pydantic
  pydantic-settings
  pyyaml
  httpx
  structlog

checkpoint:
  langgraph-checkpoint-sqlite
  langgraph-checkpoint-postgres
  psycopg
  sqlalchemy
  alembic

cli:
  typer
  rich

api:
  fastapi
  uvicorn

rag:
  langchain-postgres
  langchain-text-splitters
  pgvector

mcp:
  langchain-mcp-adapters

background:
  redis
  apscheduler

observability:
  opentelemetry-api
  opentelemetry-sdk

dev:
  pytest
  pytest-asyncio
  ruff
  pyright
```

具体版本通过 `uv.lock` 固定，不在业务代码中依赖未锁定版本。

---

## 8. 实现里程碑

| 阶段 | 里程碑 | 主要交付物 | 可验证结果 |
| --- | --- | --- | --- |
| 第一阶段：基本功能 | M1 | `harness/profile.py`、基础协议、Bootstrap、CLI | Knowledge Assistant 完成普通问答 |
| 第一阶段：基本功能 | M2 | Agent Loop、Graph、Tool Use、Permission、Hooks、Model Gateway | 能调用工具、产生 ToolResult 并处理人工确认 |
| 第一阶段：基本功能 | M3 | ConversationService、单进程 API、多用户 CLI、每用户 Runtime | 多个可信 CLI 用户并发使用且状态隔离 |
| 第一阶段：基本功能 | M4 | System Prompt、TodoWrite、Skill Loading、Context Compact、Memory、Error Recovery、Checkpoint | 会话可恢复，长上下文可压缩，错误可降级处理 |
| 第一阶段：基本功能 | M5 | Task System、MCP Tools、单 Agent 业务闭环 | 完成文件分析、报告保存和任务管理，不依赖 RAG |
| 第二阶段：RAG | M6 | RAG 协议、入库服务、document_search、Indexer、引用评测 | 基于知识文档回答并提供真实来源 |
| 第三阶段：Agent Teams | M7 | Subagent、`team.py`、Lead Agent、Researcher | Lead Agent 可以委派隔离的检索任务 |
| 第三阶段：Agent Teams | M8 | 进程内 MessageBus、Team Protocols、Analyst、Reviewer | Teammates 并行通信、审核和有限重做 |
| 第四阶段：生产化 | M9 | PostgreSQL、身份、SSE、Trace 和恢复 | Agent Server 可持久运行并安全隔离用户 |
| 第四阶段：生产化 | M10 | 幂等、分布式队列、RAG 文档上传与自动更新、Index Worker/Reconciler、Background/Cron、生产 MessageBus、受约束的 Autonomous Agents | 长任务可跨 Worker 执行和恢复，文档变更可最终一致地进入可检索索引 |
| 第四阶段：高级交互 | M11 | `/btw`、分支、重新生成、Steering、Rollback 和综合验收 | 完整知识工作流程稳定运行 |

建议严格按照 M1 到 M11 的顺序推进：M1～M5 完成第一阶段单 Agent 与多用户 CLI；M6 完成 RAG；M7～M8 完成 Agent Teams；M9～M11 按实际部署需求完成生产化和高级交互。

---

## 9. 测试策略

### 9.1 单元测试

- `harness/profile.py` 的 AgentProfile 校验和 Capability 装配。
- `state.py` 的 Reducer，以及 `graph.py` 的节点和路由条件。
- Tool 参数校验、Tool Use/ToolResult 配对和错误映射。
- Permission 决策、业务 PermissionRule 和 Hooks 触发顺序。
- System Prompt、Context、Context Compact 和 Memory。
- 第一阶段：Task System、Conversation 所有权和同一 Conversation 并发控制。
- RAG Chunk、元数据过滤和引用生成。
- Agent Teams 合约、Team Protocols 和任务预算。
- 第四阶段：Background Tasks、Cron 表达式、幂等和任务租约。

单元测试使用 Fake Model、Fake Embeddings、Fake Retriever、内存 Store 和进程内 MessageBus，避免依赖网络服务。

### 9.2 集成测试

- LangGraph + SQLite Checkpointer。
- PostgreSQL + pgvector 检索。
- 第四阶段：Redis/PostgreSQL 队列 + Worker 后台任务。
- 第四阶段：Redis MessageBus 的消息投递、去重和恢复。
- Subagent/Teammate 并行、失败和超时。
- 第一阶段：`interrupt()` 的正常恢复；第四阶段：崩溃边界下有副作用工具的幂等性。

### 9.3 端到端测试

- CLI 完整资料研究和报告生成流程。
- 第一阶段：多用户 CLI/API 隔离；第四阶段：API 流式输出。
- RAG 答案与来源一致性。
- Agent Teams 委派和审核闭环。
- 第一阶段：MCP Tools 调用链；第四阶段：Background Tasks 与 Cron Scheduler 调用链。
- 进程重启后的会话恢复。

### 9.4 Agent 评测

- 工具选择正确率。
- 工具参数正确率。
- 任务完成率。
- RAG Recall@K 和引用正确率。
- 无依据回答率。
- Agent Teams 路由正确率。
- Permission 误放行率和不必要审批率。
- Error Recovery 成功率。
- 平均响应时间、Token 和模型成本。

---

## 10. 长期非目标

以下内容不进入 M1～M11；只有出现明确业务需求时再单独立项：

- 完全去中心化的自治 Agent 网络。
- 同时支持多个向量数据库。
- Kubernetes 和复杂微服务拆分。
- Coding Agent 的 Git Worktree 隔离。
- 医疗和心理场景的专业合规实现。
- 可视化工作流编辑器。
- 完整前端界面。

先通过 `knowledge_assistant` 验证 Harness 契约和装配边界稳定。开发其他业务场景时复制本模板并替换业务实现，避免在同一项目中引入多业务 Agent 的运行时选择机制。

---

## 11. 复用模板开发其他业务 Agent

核心能力稳定后，新业务项目最少只需要替换 Profile、System Prompt 和业务 Tools：

```text
src/business/<business_agent>/
├── __init__.py             # 导出 AgentProfile
├── profile.py              # 装配模型、Prompt、Tools、Permission 和 Capabilities
├── system_prompt.py        # 业务 System Prompt
└── tools/                  # 业务工具
```

以下内容按业务需要增加：

```text
├── state.py                # 需要扩展通用 AgentState 时增加
├── schemas.py              # 有独立业务数据结构时增加
├── context.py              # 有业务上下文或 RAG 过滤条件时增加
├── permission_rules.py     # 有业务权限规则时增加
├── agent_teams/            # 启用 Agent Teams 时增加
└── knowledge/              # 启用本地知识库或 RAG 时增加
```

替换完成后，让 `entrypoints/bootstrap.py` 直接导入该项目唯一的 `AgentProfile` 并创建统一 Agent Loop；业务代码不得自行复制 Agent Loop、Permission Pipeline、Task System 或 RAG Pipeline。Agent Teams 放在该业务目录内部，由 Graph 编排，不作为多个对外业务 Agent 注册。

只有出现真正可跨业务复用的新能力时，才修改 `harness/capabilities/`；只有需要新的模型、数据库、消息总线或共享外部系统实现时，才修改 `services/`。业务需求本身不应该导致 Agent Loop 频繁变化。
