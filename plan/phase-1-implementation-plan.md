# 第一阶段实施计划：工程骨架与单 Agent 基本能力

阶段实施中发现的问题单独记录在
[Phase 1 问题记录](./phase-1-issues.md)，避免问题分析和实施步骤混在一起。

阶段状态：**M1～M5 自动化与 CLI 手工验收全部通过，第一阶段已完成（2026-08-27）**。

## 1. 阶段目标

第一阶段交付一个不依赖 RAG 和 Agent Teams 的 Knowledge Assistant 单 Agent MVP，验证通用 Harness 可以稳定完成：

```text
用户输入
  ↓
Agent Loop
  ↓
System Prompt + Context
  ↓
模型判断是否调用 Tool
  ├── 否 → 输出回答
  └── 是 → Permission → Tool → ToolResult → 再次调用模型
```

阶段结束时需要具备以下能力：

- AgentProfile、Bootstrap 和 CLI 装配链路。
- Agent Loop、Tool Use、Permission 和 Hooks。
- ConversationService、单进程 Agent Server 和多用户 CLI 客户端。
- 不同可信用户的 Conversation、Checkpoint 和 Artifact 隔离，以及共享 Model Gateway。
- TodoWrite、Skill Loading、Context Compact 和 Memory。
- Error Recovery、Checkpoint 和断点恢复。
- Task System。
- MCP Tools 接入。
- Knowledge Assistant 的文件分析、报告生成和任务管理闭环。

第一阶段明确不实现：

- RAG、Embedding、pgvector 和 `document_search` 的实际检索能力。
- Subagent、Agent Teams、MessageBus 和 Team Protocols。
- Autonomous Agents。
- 正式身份认证、SSE/WebSocket、生产 Worker、分布式队列和前端界面。
- Background Tasks、Cron Scheduler 和多 Worker 部署。
- Coding Agent 的 Worktree Isolation。

### 1.1 延期功能优先级

第一阶段不实现的能力按照依赖关系排入后续计划，而不是作为无期限 Backlog：

| 优先级 | 进入阶段 | 功能 | 排序原因 |
| --- | --- | --- | --- |
| 1 | 第二阶段 M6 | RAG、Embedding、pgvector、Indexer、引用评测 | 直接增强 Knowledge Assistant 的核心知识能力，且不依赖 Multi-Agent |
| 2 | 第三阶段 M7～M8 | Subagent、Agent Teams、进程内 MessageBus、Team Protocols、Reviewer | 依赖稳定的单 Agent、Conversation 和 RAG 契约 |
| 3 | 第四阶段 M9 | PostgreSQL Checkpoint/Store、正式身份、租户隔离、SSE、Trace、远程取消 | 是从本地可信多用户进入可部署服务的基础 |
| 4 | 第四阶段 M10 | `message_id`、副作用幂等、多 Worker、分布式队列、Agent Worker、Background Tasks、Cron、生产 MessageBus、受约束的 Autonomous Agents | 只有长任务和水平扩展出现后才有收益，并依赖持久化、租约与幂等 |
| 5 | 第四阶段 M11 | `/btw`、分支、重新生成、Steering、Rollback、跨设备恢复、前端 | 属于体验和高级交互，不阻塞核心业务闭环 |
| 6 | 未来业务扩展 | Coding Agent Worktree Isolation、完全去中心化的自治 Agent 网络 | 与当前 Knowledge Assistant 无直接依赖，不提前建设 |

本表中的“正式身份”不包含 M3 的可信本地 `--user`。M3 的用户标识只用于本地调试、
Conversation 所有权检查和 Artifact 目录隔离，不提供真实安全边界。

---

## 2. 实施顺序

第一阶段拆成五个连续里程碑：

```text
M1 工程骨架与装配
  ↓
M2 Agent Loop 与安全工具执行
  ↓
M3 Conversation Loop 与多用户 CLI
  ↓
M4 上下文、记忆与恢复
  ↓
M5 Task、MCP 与单 Agent 业务闭环
```

| 里程碑 | 主要能力 | 完成标志 |
| --- | --- | --- |
| M1 | Profile、协议、Bootstrap、CLI | Knowledge Assistant 完成普通问答 |
| M2 | Agent Loop、Tool Use、Permission、Hooks | 能安全调用工具并处理人工审批 |
| M3 | ConversationService、单进程 API、多用户 CLI、每用户 Runtime | 多个可信 CLI 用户并发使用且状态隔离 |
| M4 | System Prompt、TodoWrite、Skill、Compact、Memory、Recovery、Checkpoint | 会话可恢复，长上下文可压缩，错误可恢复 |
| M5 | Task System、MCP、业务闭环 | 完成文件分析、报告保存和任务跟踪 |

M1～M5 必须顺序完成；后一个里程碑不能通过绕过前一个里程碑的接口直接实现。

---

## 3. 统一验证约定

### 3.1 测试目录

测试代码建议放在 `src/` 之外：

```text
tests/
├── unit/                               # 不访问网络和外部服务
│   ├── harness/
│   ├── capabilities/
│   └── business/
├── integration/                        # SQLite、文件系统、Mock MCP 等集成测试
├── e2e/                                # CLI 完整场景
└── fixtures/
    ├── skills/                         # 测试 Skill
    ├── documents/                      # 第一阶段本地测试文档
    ├── mcp/                            # Mock MCP Server 数据
    └── conversations/                  # 固定模型响应和长会话数据
```

### 3.2 测试替身

默认自动化测试不得依赖真实模型和公网：

- `FakeModel`：按照预设顺序返回 Text、ToolUse、错误或超长响应。
- `FakeTool`：记录调用参数，可配置成功、失败、超时和副作用。
- `FakeClock`：控制模型网关退避时间。
- `InMemoryStore`：测试 Memory 和 Task System。
- `InMemorySaver`：测试普通图执行和 `interrupt()`。
- `MockMCPServer`：测试工具发现、组装和调用。

真实 Claude 调用只作为手工或带 `live_model` 标记的冒烟测试，不进入默认 CI。

### 3.3 统一命令

以下命令作为实现后的统一验证入口：

```bash
uv run ruff check src tests
uv run pyright src
uv run pytest -m "not live_model"
uv run pytest -m live_model
uv run agent serve
uv run agent chat --user alice
```

其中 `live_model` 测试必须显式提供 API Key 才执行；没有 API Key 时应自动跳过，而不是失败。

### 3.4 每个里程碑的完成条件

每个里程碑必须同时满足：

1. 对应代码只放在总体架构规定的目录中。
2. 单元测试通过。
3. 相关集成测试通过。
4. Ruff 和 Pyright 通过。
5. 至少完成一个 CLI 手工验证场景。
6. 错误路径与正常路径都被验证。
7. 没有通过业务代码复制 Harness 通用机制。

---

## 4. M1：工程骨架与装配

### 4.1 目标

建立可运行的项目骨架，让 Knowledge Assistant 通过统一装配链路完成一次不调用工具的模型问答。

### 4.2 涉及文件

```text
src/
├── harness/
│   ├── __init__.py
│   ├── profile.py
│   ├── agent_loop.py
│   ├── state.py
│   ├── messages.py
│   ├── graph.py
│   └── model.py
├── business/knowledge_assistant/
│   ├── __init__.py
│   ├── profile.py
│   ├── state.py
│   ├── schemas.py
│   ├── system_prompt.py
│   └── context.py
├── services/
│   ├── config.py
│   └── models.py
└── entrypoints/
    ├── bootstrap.py
    └── cli.py
```

### 4.3 实现任务

#### M1-1 项目初始化

- 创建 `pyproject.toml` 和 uv 依赖分组。
- 配置 Python 3.12、Ruff、Pyright 和 pytest。
- 使用 `pydantic-settings` 定义模型、环境和日志配置。
- 禁止在代码中硬编码 API Key、模型 ID 和数据库地址。

#### M1-2 通用契约

在 `harness/profile.py` 定义：

- `AgentProfile`
- Model 配置引用
- System Prompt Provider
- Tool 列表
- PermissionRule 列表
- ContextProvider 列表
- Capability 列表

在 `state.py` 和 `messages.py` 定义：

- `AgentState`
- Message
- ToolUse
- ToolResult
- 状态 Reducer

业务 Profile 只能组合这些契约，不能包含 Agent Loop 实现。

#### M1-3 最小 Agent Loop

- `graph.py` 先实现仅包含 Model Node 的最小 StateGraph。
- `agent_loop.py` 提供 `invoke()`、`ainvoke()` 和 `stream()` 入口。
- 本里程碑不实现 Tool Node 和 Permission Node。
- `model.py` 只定义模型协议；Anthropic 适配放在 `services/models.py`。

#### M1-4 业务装配

- `knowledge_assistant/system_prompt.py` 提供最小业务 Prompt。
- `knowledge_assistant/profile.py` 创建 Knowledge Assistant Profile。
- `entrypoints/bootstrap.py` 直接使用唯一业务 Profile 创建 Model 和 Agent Loop。
- `entrypoints/cli.py` 只负责读取输入、调用 Agent Loop 和输出结果。

### 4.4 自动化验证

#### 单元测试

- 缺少模型配置或 System Prompt 时，Profile 校验失败。
- Bootstrap 使用 FakeModel 构建 Agent Loop，不依赖 Anthropic SDK。
- Knowledge Assistant Profile 不导入 `services/models.py`。
- State Reducer 能正确追加消息且不覆盖原状态。

#### 集成测试

使用 FakeModel 返回固定文本：

```text
输入：请介绍你能做什么
输出：固定的 Knowledge Assistant 能力说明
```

验证调用链：

```text
CLI → Bootstrap → Profile → AgentLoop → Graph → FakeModel
```

### 4.5 手工验证

```bash
uv run agent chat
```

输入一个不需要工具的问题，确认：

- CLI 能一次性输出回答；Token 级流式输出不属于 M1。
- 日志中包含 `agent_name` 和 `thread_id`。
- 业务层没有直接创建 Anthropic 客户端。
- 切换模型配置不需要修改业务 Profile 代码。

### 4.6 M1 验收标准

- Knowledge Assistant 可以完成普通问答。
- Profile、Bootstrap、Agent Loop 职责分离。
- FakeModel 自动化测试全部通过。
- 真实模型冒烟测试可以通过配置启用。
- 尚未出现 Tool Use、RAG 或 Agent Teams 代码路径。

---

## 5. M2：Agent Loop 与安全工具执行

### 5.1 目标

把最小模型调用扩展成完整 Agent Loop，并实现 Tool Use、Permission Pipeline、Hooks 和人工审批。

### 5.2 涉及文件

```text
src/
├── harness/
│   ├── agent_loop.py
│   ├── state.py
│   ├── messages.py
│   ├── graph.py
│   ├── tool_use.py
│   ├── permissions.py
│   └── hooks.py
├── business/knowledge_assistant/
│   ├── profile.py
│   ├── permission_rules.py
│   └── tools/
│       ├── calculator.py
│       ├── file_reader.py
│       └── report_writer.py
└── services/
    ├── artifacts.py
    ├── model_gateway.py
    ├── security.py
    └── observability.py
```

### 5.3 实现任务

#### M2-1 完整 Agent Loop

在 `graph.py` 中实现：

- Model Node
- Tool 路由条件
- Permission Node
- Tool Node
- ToolResult 回注
- 最终输出节点
- 最大迭代次数和取消信号

`agent_loop.py` 只作为调用门面，不重复实现节点和路由。

#### M2-2 Tool Use

在 `tool_use.py` 中实现：

- Tool 协议和参数 Schema。
- Tool Registry 和 dispatch。
- ToolUse ID 与 ToolResult 配对。
- 多 ToolUse 的并发安全分组。
- Tool 超时、异常映射和输出截断。
- 未知 Tool 的明确错误结果。

首批业务工具：

- `calculator`：只允许受限表达式，不调用 shell `eval`。
- `file_reader`：只读取授权目录，防止路径穿越。
- `report_writer`：写入 Artifacts 目录，覆盖文件需要审批。

#### M2-3 Permission Pipeline

实现状态：已完成。已覆盖四态决策、规则顺序、默认拒绝、业务规则、
`InMemorySaver`、`interrupt()` 暂停，以及同一 `thread_id` 的批准/拒绝恢复。

在 `permissions.py` 中实现：

- `ALLOW`
- `DENY`
- `ASK`
- `PASSTHROUGH`
- PermissionRule 执行顺序和结果合并。
- 默认拒绝未知高风险操作。
- `ASK` 通过 LangGraph `interrupt()` 暂停。

业务规则放在 `knowledge_assistant/permission_rules.py`：

- 知识目录文件自动允许、工作区内其他文件询问授权、工作区外文件拒绝。
- 报告写入和覆盖规则。
- 外部发布默认禁止。

M2 使用 InMemorySaver 验证暂停和恢复；SQLite 持久恢复在 M3 完成。

#### M2-4 Hooks

实现状态：已完成。已覆盖四类 Hook 事件、注册顺序、PreToolUse 阻断、
异常继续/中止策略、脱敏 Tool 日志、大输出告警和 Stop 单次统计。

在 `hooks.py` 中实现：

- `UserPromptSubmit`
- `PreToolUse`
- `PostToolUse`
- `Stop`
- Hook 注册、执行顺序和阻断结果。

首批 Hook：

- Tool 调用日志。
- Permission 检查桥接。
- 大输出告警。
- Stop 统计。

#### M2-5 Model Gateway

实现状态：已完成。

在 `services/model_gateway.py` 中实现：

- 使用额度范围和文件锁限制本机多个 CLI 进程的模型并发。
- 主模型达到并发上限时，选择具有独立额度范围的备用模型。
- HTTP 429 使用指数退避与随机抖动重试一次，仍失败时切换备用模型。
- 超时、连接错误和 5xx 使用指数退避与随机抖动，最多重试两次。
- SDK 内部重试关闭，由 Model Gateway 统一计数并展示。
- 重试等待期间释放并发槽。
- 主模型耗尽临时错误重试后切换备用模型。
- CLI 使用 `You [conversation 后六位]:` 标识用户输入，并使用 `agent_name [conversation 后六位] [实际模型]:` 标识每轮回答；发生重试或降级时，另行展示重试次数、等待时间、降级模型和降级原因。
- 所有模型均失败时，503 响应携带本次重试和降级事件，CLI 展示过程及最终失败原因。

备用模型未配置时不会伪造降级；所有路由不可用时返回明确的
`ModelGatewayUnavailableError`。

### 5.4 自动化验证

#### Agent Loop 测试

FakeModel 依次返回：

```text
第一次：ToolUse(calculator, {expression: "2 + 3"})
第二次：Text("结果是 5")
```

验证：

- Tool 只执行一次。
- ToolUse ID 与 ToolResult ID 一致。
- ToolResult 被添加到消息后模型才进行第二次调用。
- 模型返回纯文本后循环结束。
- 超过最大迭代次数时安全退出。

#### Permission 测试

| 场景 | 期望结果 |
| --- | --- |
| 读取知识目录文件 | `ALLOW` |
| 读取工作区内、知识目录外文件 | `ASK` |
| 读取工作区外文件 | `DENY` |
| 新建报告草稿 | `ALLOW` 或按业务规则处理 |
| 覆盖已有正式报告 | `ASK` |
| 未知危险工具 | `DENY` |
| 当前规则不适用 | `PASSTHROUGH` 并继续后续规则 |

验证用户批准后只执行一次，拒绝后生成 ToolResult 并让模型继续解释拒绝原因。

#### Hooks 测试

- Hook 按注册顺序执行。
- `PreToolUse` 可以阻断工具。
- `PostToolUse` 能读取工具结果但不能重复执行工具。
- `Stop` 在最终退出前触发一次。
- Hook 异常被记录并按照配置决定中止或继续。

#### Model Gateway 测试

- 前两次超时、第三次成功，验证非 429 最多重试两次。
- 主模型连续两次返回 429，验证第一次显示 `1/1`，第二次失败后降级。
- 验证退避延迟为指数增长并包含随机抖动。
- 主模型并发槽占用时调用备用模型。
- 主模型耗尽临时错误重试后调用备用模型。
- 同步和异步调用使用相同策略。
- CLI 输出重试编号和模型降级信息。

#### 安全测试

- `../`、绝对路径、符号链接不能绕过文件范围检查。
- Calculator 拒绝函数导入、属性访问和任意代码执行。
- Tool 输出超过限制时被截断并保留 Artifact 引用。
- 日志不得输出 API Key 和完整敏感文件内容。

### 5.5 手工验证

依次执行以下场景：

1. “计算 `(15 + 5) * 3`。”——直接允许并返回 60。
2. “读取 fixtures 目录中的 sample.txt。”——允许读取。
3. “读取工作区外的文件。”——触发审批或拒绝。
4. “生成 report.md。”——按业务规则写入。
5. “覆盖已有 report.md。”——触发人工确认。

### 5.6 M2 验收标准

- 单次和多个 ToolUse 均能正确执行。
- Permission 四种结果均有自动化测试。
- `interrupt()` 可以在同一线程中批准、拒绝和恢复。
- Hooks 不侵入 Agent Loop 主流程。
- 所有文件工具通过路径安全测试。
- Tool 失败不会导致消息结构损坏。

### 5.7 M2 手工验收记录

验收状态：**通过（2026-08-23）**。

| 场景 | 实际结果 |
| --- | --- |
| Calculator 计算 `(15 + 5) * 3` | 自动允许并返回 `60` |
| 读取知识目录 `sample.txt` | 无授权提示，返回 `M2 文件读取验收通过` |
| 读取工作区内、知识目录外文件 | 触发 `ASK`；批准后读取，拒绝后不执行 |
| 读取工作区外文件 | 直接 `permission_denied`，不提供授权入口 |
| 新建 `artifacts/report.md` | 自动允许，文件内容写入成功 |
| 覆盖 `artifacts/report.md` | 触发 `ASK`；批准后内容更新成功 |
| Kimi 达到本地并发上限 | CLI 显示降级信息，DeepSeek `deepseek-v4-flash` 返回 `OK` |

验收期间发现的 Checkpoint 类型白名单警告记录为
[ISSUE-P1-002](./phase-1-issues.md#issue-p1-002checkpoint-自定义消息类型白名单警告)，
不影响本次权限结果，但需要在 M3 持久化 Checkpoint 前处理。

---

## 6. M3：Conversation Loop 与多用户 CLI

### 6.1 目标

在现有 Agent Loop 外增加多轮 Conversation 层，并通过单进程 Agent Server 支持多个
可信 CLI 用户同时使用。第一阶段只解决本地开发环境中的会话隔离和异步并发，不把
`--user` 当成正式身份认证。

```text
CLI Alice ──┐
CLI Bob   ──┼──→ FastAPI Agent Server
CLI Carol ──┘              ↓
                    UserRuntimeRegistry
                    ├─ Alice AgentLoop + InMemorySaver
                    ├─ Bob AgentLoop + InMemorySaver
                    └─ Carol AgentLoop + InMemorySaver
                              ↓
                       共享 ModelGateway
```

### 6.2 涉及文件

```text
src/
├── harness/
│   └── conversation.py          # ConversationService 和 Run 结果
├── services/
│   ├── checkpoint.py            # Checkpointer 工厂
│   └── observability.py         # Run 级 Model Gateway 事件隔离
└── entrypoints/
    ├── bootstrap.py             # 共享服务和每用户 Runtime Registry
    ├── api.py                   # 单进程 FastAPI Agent Server
    └── cli.py                   # 多用户 HTTP CLI 客户端
```

新增直接依赖：

- FastAPI：单进程 Agent Server。
- Uvicorn：本地 ASGI Server。
- HTTPX：CLI HTTP Client。
- HTTPX ASGITransport：不启动真实端口的 API 自动化测试。

### 6.3 实现任务

#### M3-1 Conversation 契约

- `conversation_id` 直接作为 LangGraph `thread_id`。
- 一次用户输入到最终回答使用一个应用级 `run_id`。
- Permission interrupt/resume 沿用原 `conversation_id/thread_id/run_id`。
- `ConversationService` 只处理一条消息或一次恢复，不读取终端输入。
- 本阶段不增加 `turn_id`、`message_id`、幂等记录或 Run 数据库表。

#### M3-2 Checkpointer 注入

- 从 `harness/graph.py` 删除硬编码的 `InMemorySaver()`。
- `services/checkpoint.py` 创建 Checkpointer，并通过 Bootstrap 注入 AgentLoop。
- 本里程碑继续使用 `InMemorySaver`；SQLite 重启恢复在 M4 完成。
- 每个用户的 Runtime 在首次访问时创建，此后复用，不按消息重复编译 Graph。

#### M3-3 单进程 Agent Server

实现最小接口：

```text
POST /users/{user_id}/conversations
GET /users/{user_id}/conversations
DELETE /users/{user_id}/conversations/{conversation_id}
POST /users/{user_id}/conversations/{conversation_id}/messages
POST /users/{user_id}/conversations/{conversation_id}/runs/{run_id}/permission
```

- 使用进程内 Registry 保存 Conversation Owner、状态和活跃 `run_id`。
- 列表接口只返回当前用户拥有的 Conversation。
- 删除接口同时清理 Conversation Registry 和对应的 LangGraph Checkpoint。
- `running` 或 `waiting_permission` 状态的 Conversation 不允许删除。
- 不同 Conversation 可以异步执行。
- 同一 Conversation 已有 `running` 或 `waiting_permission` Run 时拒绝普通新消息。
- Permission 请求返回客户端，不占用执行中的模型请求；审批后通过 Resume 接口继续。
- API 返回本次 Run 收集到的模型重试和降级事件，不实现实时流式传输。

#### M3-4 多用户 Runtime 与 Artifact 隔离

- 多个用户共享 ModelGateway，使本地并发额度、重试和备用模型路由统一生效。
- 每个用户复用自己的 AgentLoop、Checkpointer 和业务工具装配。
- 公共 Knowledge Root 和内置 Skill 共享；每个用户另有独立的 Workspace、私有 Knowledge、
  私有 Skill 和 Artifact Root。
- 用户目录默认为 `users/<user_id>/{workspace,knowledge,skills,memory}`，Artifact 使用
  `artifacts/<user_id>/`。
- `file_reader` 只允许当前用户 Workspace、当前用户私有 Knowledge、公共 Knowledge 和当前用户
  Artifact；即使批准也不能读取另一用户的目录。
- 模型客户端和 ModelGateway 全局共享，但真实 Provider usage 按请求收集并累计到用户级内存账本；
  CLI 使用 `/usage` 查看当前用户用量。
- `user_id` 只接受受限字符并经过路径安全校验，不能直接形成任意文件路径。
- Conversation Owner 不匹配时返回 `403 Forbidden`。

#### M3-5 CLI Client

- `agent serve` 启动单进程 Agent Server。
- `agent chat --user <user_id>` 连接 Server 并创建 Conversation。
- `/new` 创建新 Conversation。
- `/list` 列出当前用户的 Conversation，并标记当前会话。
- `/switch <ID>` 使用完整 ID 或唯一 ID 后缀切换到空闲 Conversation。
- `/delete <ID>` 确认后删除空闲 Conversation；删除当前会话后自动创建新会话。
- `/exit` 和 `/quit` 退出。
- 普通回答完成后继续读取下一条输入。
- Permission 请求由 CLI 展示，用户选择后调用 Resume API。
- 保留单次消息入口，但通过同一 API 和 ConversationService 执行。

### 6.4 自动化验证

- Alice 与 Bob 的消息历史、Checkpoint 和 Artifact 目录相互隔离。
- Alice 与 Bob 的 Workspace、私有 Knowledge、私有 Skill 和 Token 用量相互隔离。
- Alice 读取自己的私有 Knowledge 成功，Bob 使用相同绝对路径时得到 `permission_denied`。
- FakeModel 延迟响应时，不同 Conversation 可通过 `asyncio.gather()` 并发执行。
- 同一 Conversation 的第二个活跃请求返回 `409 Conflict`。
- Bob 访问 Alice 的 Conversation 或提交审批时返回 `403 Forbidden`。
- Permission interrupt/resume 使用相同 `conversation_id/thread_id/run_id`。
- 两个并发 Run 的 ModelGateway 事件不会交叉。
- 两个并发 Run 的模型 usage 不会交叉，用户累计 Token 用量分别记账。
- 连续发送两条消息时，第二个模型请求能读取同一 Thread 的第一轮历史。
- `/new` 后的新 Conversation 不继承旧 Conversation 的消息。
- `/list` 不显示其他用户的 Conversation。
- `/switch` 能恢复目标 Conversation 原有的 Checkpoint 和消息历史。
- `/delete` 后 Conversation 不可访问且 Checkpoint 已清除；活跃 Conversation 返回 `409`。
- 所有 API 集成测试使用 ASGI Test Client、FakeModel 和 InMemorySaver，不访问公网。

### 6.5 手工验证

启动 Server：

```bash
uv run agent serve
```

分别在两个终端运行：

```bash
uv run agent chat --user alice
uv run agent chat --user bob
```

确认：

- Alice 和 Bob 可以同时提问。
- 两人的上下文不会串线。
- Alice 的 Artifact 写入 `artifacts/alice/`，Bob 写入 `artifacts/bob/`。
- Alice 的私有资料和 Skill 放入 `users/alice/`，Bob 无法在 Prompt 或 ToolResult 中看到。
- 一个用户触发权限等待时，另一个用户仍能完成普通问答。
- 模型重试和降级信息只显示在对应 CLI。
- 使用 `/new`、`/list`、`/switch` 在多个 Conversation 间切换，历史互不串线。
- 使用 `/delete` 删除非当前会话和当前会话，确认当前会话删除后 CLI 自动新建会话。
- 使用 `/usage` 查看当前用户在本次 Server 进程中的累计 Token 用量。

### 6.6 M3 验收标准

- 单进程 Agent Server 可以服务至少两个同时在线的可信 CLI 用户。
- 每用户 Runtime 只创建一次，用户消息不会重复编译 Graph。
- Conversation Owner、状态和权限恢复校验全部通过。
- 不同 Conversation 并发、同一 Conversation 拒绝并发的行为稳定。
- 本里程碑不依赖 Redis、PostgreSQL、后台 Worker 或正式身份系统。

### 6.7 M3 自动化验收记录

验收状态：**自动化与手工双终端验收通过（2026-08-24）**。

- FastAPI ASGI 测试验证 Alice/Bob 的 Runtime、消息历史和 Artifact Root 相互隔离。
- 不同 Conversation 并发执行通过；同一 Conversation 的第二个活跃 Run 返回 `409`。
- Conversation Owner 不匹配返回 `403`，非法 `user_id` 返回 `422`。
- Conversation 列表、切换和删除通过；删除同步清理 Checkpoint，活跃会话拒绝删除。
- Permission 恢复沿用原 `conversation_id/thread_id/run_id`。
- 两个并发 Run 的 ModelGateway 事件通过 `ContextVar` 隔离。
- 公共资料/内置 Skill 共享，用户 Workspace、私有 Knowledge/Skill 和 Artifact 已完成目录隔离。
- `file_reader` 跨用户绝对路径读取被拒绝；Token usage 已通过 `ContextVar` 收集并按用户记账。
- `LANGGRAPH_STRICT_MSGPACK=true` 下完整测试套件通过，ISSUE-P1-002 已关闭。
- 默认自动化测试没有访问公网或调用真实模型。

---

## 7. M4：上下文、记忆与恢复

### 7.1 目标

实现长会话需要的 System Prompt、TodoWrite、Skill Loading、Context Compact、Memory、Checkpoint 和 Error Recovery。

### 7.2 涉及文件

```text
src/
├── harness/
│   ├── context.py
│   ├── system_prompt.py
│   ├── error_recovery.py
│   └── capabilities/
│       ├── todo_write.py
│       ├── skill_loading.py
│       ├── context_compact.py
│       └── memory.py
├── business/knowledge_assistant/
│   ├── profile.py
│   ├── system_prompt.py
│   └── context.py
└── services/
    ├── checkpoint.py
    └── stores.py
```

### 7.3 实现任务

#### M4-1 System Prompt 与 Context

- `harness/system_prompt.py` 按稳定顺序组装基础说明、工具说明、业务片段、Skill 摘要、Memory 和运行时上下文。
- `harness/context.py` 负责上下文预算、选择和去重。
- `business/.../system_prompt.py` 只包含知识助手业务说明。
- `business/.../context.py` 只提供当前任务和允许访问的本地资料上下文。
- Prompt 中不得直接注入密钥、内部对象或未经隔离的完整文件。

实现状态：**自动化与 CLI 手工验收均通过（2026-08-27）**。

- `SystemPromptBuilder` 已按固定六段顺序组装完整 Prompt，并为 Skill 和 Memory 保留后续接入位置。
- `ContextManager` 已实现优先级稳定排序、按 key/内容去重、Fragment 数量限制和字符预算截断。
- Knowledge Assistant Context 只注入最近用户任务和授权知识目录的相对文件名。
- 业务 Context 不读取文件正文，不注入隐藏文件、绝对目录、配置对象或模型凭据。
- Agent Graph 的同步和异步 Model Node 均通过同一 Prompt Builder 创建 `ModelRequest`。

#### M4-2 TodoWrite 与 Skill Loading

- Todo 状态包含内容、状态和当前步骤。
- 同一时间最多一个 Todo 为 `in_progress`。
- Skill 启动时只扫描名称和描述。
- 只有模型明确调用 Skill 时才加载正文。
- Skill 不得覆盖 System Prompt 和 Permission。

实现状态：**自动化与 CLI 手工验收均通过（2026-08-27）**。

- `TodoWriteTool` 采用整表替换语义，将内容、状态和当前步骤写入当前 Thread 的
  `capability_state.todo_write`；不同 Conversation 不共享 Todo。
- Todo 输入最多 20 项，且通过 Pydantic 校验保证最多一个 `in_progress`。
- `SkillCatalog` 启动时只扫描 `SKILL.md` 的 YAML frontmatter，并将名称和描述注入
  System Prompt；完整正文仅由 `load_skill` 按 Registry 名称读取。
- `load_skill` 不接受文件路径，并限制 Skill 数量、frontmatter 大小、正文大小和根目录范围。
- TodoWrite 和 Skill Loading 都注册了独立 PermissionRule；Skill 内容被标记为辅助指导，不能
  覆盖 System Prompt、当前用户指令或 Tool Permission。
- Knowledge Assistant 内置 `knowledge-synthesis` 调试 Skill，业务 Profile 已装配两项能力。
- 手工验证公共 `knowledge-synthesis` 与 Alice 私有 `alice-workflow` 同时可见，私有 Skill
  可按需加载；用户间 Skill、Knowledge、文件和 Artifact 边界符合预期。TodoWrite 已在
  M5-3 最终业务闭环中随任务规划和收尾一并验收。

#### M4-3 Context Compact

按成本从低到高实现：

1. 删除冗余状态和重复消息。
2. 裁剪旧 ToolResult，并把大结果保存为 Artifact。
3. 保留最近工具交互。
4. 对旧历史生成摘要。
5. Prompt 过长时执行 Reactive Compact 后重试一次。

压缩后必须保留：

- 当前任务目标。
- 未完成 Todo 和 Task。
- 最近一次 ToolUse/ToolResult 配对。
- Permission 审批结果。
- 关键来源和文件引用。

实现状态：**自动化与 CLI 手工验收均通过（2026-08-24）**。

- Graph 在每次 Model Node 前执行分层 Compact：相邻重复普通消息去重、大型 ToolResult 落盘、
  旧 ToolResult Micro Compact，超过预算后再调用模型生成历史摘要。
- 压缩后的消息使用 LangGraph `Overwrite` 替换 Reducer 中的旧历史，不会把已压缩消息再次追加。
- 摘要前的完整消息以 JSONL 保存到当前用户的 `artifacts/<user_id>/transcripts/`；大型结果保存到
  `artifacts/<user_id>/tool-results/`，当前用户可通过 `file_reader` 安全回读，其他用户不可访问。
- 摘要 Prompt 明确保留当前目标、用户约束、来源、Artifact 引用、关键结论、错误和未完成工作；
  当前 Todo、Task 和 Permission 历史另以受保护运行状态注入摘要消息。
- 主模型返回常见 Prompt/Context 过长错误时，执行一次 Reactive Compact 并补做一次模型请求；
  第二次仍失败会直接终止，不形成无限重试。
- 阈值通过 `AGENT_CONTEXT_COMPACT__*` 配置；当前用字符数作跨模型保守估算，不把它冒充为
  Provider 的精确 Token 数。
- CLI 手工验证确认：大型 ToolResult 正确落盘、完整 Transcript 可检查、压缩后仍保留当前目标
  和 Permission 结果、当前用户可回读自己的 Artifact，其他用户无法跨目录读取。

#### M4-4 Memory

- 区分用户偏好、用户反馈、项目事实和参考资料。
- 支持写入、索引、选择性读取和合并。
- Memory 跨会话存在，但不能自动覆盖当前用户明确指令。
- 不同用户或租户的 Memory 必须隔离。

实现状态：**自动化与 CLI 手工验收均通过（2026-08-24）**。

- Memory 类型沿用 `learn-claude-code/s09` 的 `user`、`feedback`、`project` 和 `reference`；
  分别承载用户偏好、做事反馈、项目事实和资料位置。
- 每条 Memory 保存为当前用户 `users/<user_id>/memory/<name>.md`，采用 YAML frontmatter；
  `MEMORY.md` 是低成本索引，`.history/<name>.jsonl` 保存每次写入的来源、ToolUse ID 和完整版本。
- `memory_write` 使用稳定 name 执行新建或同名合并更新；持久化会影响后续 Conversation，
  因此每次都通过 Permission `ASK` 请求确认。`memory_search` 只能读取当前用户 Store，可自动允许。
- System Prompt 始终只放受预算约束的索引，并使用中英文关键词评分选择最多 5 条相关正文；
  无关 Memory 的完整内容不会进入当前请求，也不会额外消耗一次模型 side-query。
- Memory Store 位于用户 Runtime 而不是 Conversation Checkpoint，所以同一用户可以跨 Conversation、
  跨 Server 重启读取；Alice 和 Bob 的 Memory Root、Tool 和 Prompt 注入相互隔离。
- 基础 Prompt 明确规定当前用户的显式请求高于历史 Memory，Memory 只能作为辅助上下文。
- 当前阶段采用“主 Agent 显式调用 Tool + 用户确认”的写入方式，不在每轮结束后静默调用模型提取；
  这样避免额外 Token 消耗和未经确认的长期信息保存。自动提取可在未来作为默认关闭的选装能力。
- CLI 手工验证确认：写入前正确请求 Permission、同名更新产生新 revision 且保留审计历史、
  新 Conversation 和 Server 重启后可以恢复、当前明确指令能够覆盖历史偏好、拒绝写入不落盘，
  且 Bob 无法检索 Alice 的 Memory。

#### M4-5 Checkpoint

- 本地使用 SQLite Checkpointer。
- 所有运行使用稳定的 `thread_id`。
- 保存消息、Todo、任务引用、审批状态和压缩状态。
- 进程重启后从最后一个安全节点恢复。
- 第一阶段只保证从已保存 Checkpoint 恢复，不承诺进程在副作用提交瞬间崩溃时的 Exactly-Once；工具幂等进入第四阶段 M10。

实现状态：**自动化与 CLI 手工验收通过（2026-08-25）**。

- 使用官方 `langgraph-checkpoint-sqlite`，每个用户保存到
  `users/<user_id>/checkpoints.sqlite3`；`conversation_id` 继续直接作为稳定 `thread_id`。
- `LocalSqliteSaver` 在同一个带锁的 SQLite 连接上兼容同步 CLI 测试和异步 HTTP AgentLoop，
  开启 WAL、`busy_timeout` 和受限文件权限；`InMemorySaver` 仅保留给隔离单元测试。
- LangGraph Checkpoint 保存完整 `AgentState`，包括消息、`capability_state` 中的 Todo、Task 引用和
  Compact 状态，以及 interrupt 所需的 pending tool 状态。
- Conversation 所有权、运行状态、原 `run_id` 和 `PermissionRequest` 另存到
  `.agent/conversations.sqlite3`；否则进程重启后虽有 Graph Checkpoint，API/CLI 仍无法定位它。
- CLI `/list` 可以看到重启前的会话；`/switch <ID>` 切换到 `waiting_permission` 会话时会重新展示
  Permission 内容，并用原 `conversation_id`/`run_id` 恢复，而不是创建新 Run。
- 消息 API 支持选装 `required_tool`，用于明确业务动作和确定性验收；工具名必须属于当前 Profile，
  且只强制当前 Turn 的第一次模型调用，后续 ToolResult 总结请求恢复为 `auto`，避免重复调用。
- Kimi K3 始终开启 Thinking，当前服务会拒绝指定函数对象形式的 `tool_choice`；Moonshot 适配器在
  强制调用时只暴露目标工具并发送 `tool_choice=required`，从而保持确定性和 K3 兼容性。
- DeepSeek V4 Thinking 当前不支持 `required_tool` 所需的 Tool Choice；模型网关会跳过该不兼容
  降级路由并返回可重试的 503，不把“必须调用”静默弱化为无法保证的自动工具选择。
- 普通聊天继续使用模型默认的自动选工具模式；System Prompt 禁止模型在没有匹配 ToolResult 时
  声称外部操作已经完成。
- 自动化覆盖普通消息历史跨 Application 重建、完整 capability state SQLite round-trip、
  Permission interrupt 跨重建批准、重复 Resume 返回 409，以及 Conversation 删除同步清理 Checkpoint。
- 进程在普通模型调用中退出时，遗留的 `running` 标记会恢复为 `idle`，避免会话永久锁死；本期不自动
  重放没有明确 interrupt 输入的半完成模型请求。
- 当前仍是单进程本地实现。多进程租约、PostgreSQL Checkpointer、事务性 Outbox、工具副作用幂等和
  Exactly-Once 继续保留到第四阶段 M9～M10。

#### M4-6 Error Recovery

至少区分：

- 可重试模型错误。
- 限流和服务不可用。
- Prompt 过长。
- 输出 Token 不足。
- Tool 超时和 Tool 参数错误。
- 用户取消。

恢复策略包括退避重试、Reactive Compact、提高输出上限、切换备用模型和安全终止。

实现状态：**自动化与 CLI 手工验收均通过（2026-08-25）**。

- `harness/error_recovery.py` 提供与 Provider SDK 解耦的稳定错误分类，区分临时错误、429、
  服务不可用、Prompt 过长、输出上限、请求拒绝和取消。
- Model Gateway 对网络错误、408、429 和 5xx 使用指数退避与随机抖动；429 优先采用
  `Retry-After`，但仍受配置的最大等待时间保护。主路由耗尽后才切换备用模型。
- Prompt 过长时执行一次 Reactive Compact；重试仍失败则抛出明确错误并由 API 返回 `413`，
  不形成无限压缩或重试。
- Provider 的 `finish_reason`/`stop_reason` 会标准化保存。响应因长度截断时把输出上限从默认
  4096 提高到 8192 后重做一次；仍截断时安全终止并由 API 返回 `502`。上下限和倍率可配置。
- Tool 参数错误、超时、执行异常、未知工具和结果 ID 不一致统一返回带稳定 `error`、`message`
  和 `retryable=false` 的结构化 ToolResult。副作用工具不会被自动盲目重试。
- API 增加活跃 Run 取消入口；第二个 CLI 可以使用 `/cancel <conversation-id-suffix>` 取消正在
  执行的 Graph Task。取消后 Conversation 回到 `idle` 并可继续使用。
- 本期取消范围仍是单进程正在执行的 Run。跨进程远程取消、租约和持久化取消信号保留到 M9。

### 7.4 自动化验证

#### System Prompt 测试

- 各片段顺序固定。
- 可选片段为空时不会出现多余占位文本。
- Skill 正文在未调用时不进入 Prompt。
- 敏感配置不会出现在 Prompt 快照中。

#### Context Compact 测试

构造包含大量 ToolResult 的固定长会话，验证：

- 压缩后大小低于预算。
- ToolUse 和 ToolResult 不会失配。
- 最近消息、当前目标和未完成任务仍存在。
- 大输出被保存为 Artifact，并保留可读取引用。
- 同一轮最多执行一次 Reactive Compact，避免无限重试。

#### Memory 测试

- 会话 A 保存偏好后，会话 B 可以按需加载。
- 无关 Memory 不进入当前上下文。
- 用户 A 无法读取用户 B 的 Memory。
- 新反馈可以更新旧偏好，并保留审计来源。

#### Checkpoint 测试

1. 运行到人工审批节点后停止进程。
2. 重新创建 Bootstrap 和 Agent Loop。
3. 使用同一 `thread_id` 恢复。
4. 批准后继续执行。
5. 验证从 Permission interrupt 恢复时写入工具只执行一次。

#### Error Recovery 测试

使用 FakeModel 注入以下错误：

| 错误 | 期望恢复 |
| --- | --- |
| 临时网络错误 | 指数退避后成功 |
| 限流 | 遵守 Retry-After 或退避策略 |
| Prompt 过长 | Reactive Compact 后重试 |
| 输出上限不足 | 提高上限或发起续写 |
| 主模型持续失败 | 切换备用模型或安全终止 |
| Tool 参数错误 | 返回结构化 ToolResult，不进行盲目重试 |

### 7.5 手工验证

- 连续进行多轮文件分析，观察 Context Compact 前后的关键目标是否一致。
- 告诉 Agent 一个输出偏好，开启新会话后确认可以按需恢复。
- 在等待报告覆盖审批时强制退出，再启动 CLI 恢复。
- 使用错误模型 ID 模拟失败，确认 Error Recovery 给出明确结果而不是死循环。

### 7.6 M4 验收标准

- System Prompt 可以运行时组装且不依赖硬编码大字符串。
- Skill 正文按需加载。
- 长会话压缩后仍能继续正确完成任务。
- Memory 可以跨会话读取并保持用户隔离。
- SQLite Checkpoint 支持进程重启恢复。
- 所有恢复策略都有最大次数和明确终止条件。

---

## 8. M5：Task、MCP 与单 Agent 业务闭环

### 8.1 目标

完成 Task System 和 MCP Tools，并用 Knowledge Assistant 验证不依赖 RAG、Agent Teams、
Background Tasks 或 Cron Scheduler 的完整单 Agent 业务流程。

### 8.2 涉及文件

```text
src/
├── harness/capabilities/
│   └── task_system.py
├── business/knowledge_assistant/
│   ├── profile.py
│   └── tools/
│       ├── calculator.py
│       ├── file_reader.py
│       └── report_writer.py
├── services/
│   ├── stores.py
│   ├── artifacts.py
│   ├── observability.py
│   └── mcp_tools.py
└── entrypoints/
    └── cli.py
```

### 8.3 实现任务

#### M5-1 Task System

实现状态：**自动化与 CLI 手工验收均通过（2026-08-25）**。

- `TaskRecord` 已定义唯一 Task ID、标题、描述、`pending → in_progress → completed/failed`
  状态、依赖、Owner、结果引用、失败原因，以及运行时自动写入的 `conversation_id/run_id`
  来源字段；关联信息不暴露为模型可填写的 Tool 参数。
- `InMemoryTaskStore` 用于无数据库单元测试；`SQLiteTaskStore` 使用 WAL、`BEGIN IMMEDIATE`
  和条件更新保证跨连接并发认领时只有一个 Owner 成功。
- 每个用户使用 `users/<user_id>/tasks.sqlite3`，Task 跨 Conversation 和进程重启保留，
  Alice/Bob 的 Task 数据相互隔离。
- SQLite 启动迁移会为旧 `tasks` 表增加可空关联列；历史 Task 保留且来源为 `null`，
  新 Task 可按 Conversation/Run 直接区分同名记录。
- Knowledge Assistant 已直接注册 `create_task`、`get_task`、`list_tasks`、`claim_task`、
  `complete_task` 和 `fail_task`，并通过统一 Permission Pipeline 与 Hooks 执行。
- 自动化覆盖唯一 ID、合法状态转换、依赖阻塞、并发认领、重启恢复、配置加载、
  Profile 装配和用户隔离；默认测试不调用真实模型。

- 定义 Task ID、标题、描述、状态、依赖、Owner 和结果引用。
- 支持创建、查看、认领、完成和失败。
- 依赖未完成的 Task 不允许认领。
- Task 状态通过 Store 持久化。
- Knowledge Assistant 直接注册通用 Task System Tools，不创建业务包装文件。

#### M5-2 MCP Tools

实现状态：**自动化与 CLI 手工验收通过（2026-08-25）**。

- 使用官方 `langchain-mcp-adapters` 和 MCP Python SDK，支持 stdio、Streamable HTTP、
  SSE 与 WebSocket 配置；MCP 默认关闭，只有显式配置的 Server 才会连接。
- 工具名沿用 learn-claude-code s19 的 `mcp__server__tool` 规则，非
  `[a-zA-Z0-9_-]` 字符替换为 `_`，规范化后冲突使用稳定数字后缀。
- MCP JSON Schema 直接进入统一 `ToolDefinition` 和参数校验，调用结果、协议错误、
  连接错误与超时统一转换为项目的 `ToolResult`。
- `readOnlyHint` 工具自动允许，`destructiveHint` 工具进入 Permission interrupt；
  没有安全声明的外部工具采用保守的 ASK 策略。
- 各 Server 并发发现且独立隔离；单个 Server 连接失败或超时只产生结构化失败记录，
  不会移除健康 MCP 工具或 calculator、file_reader 等本地工具。
- MCP 工具在 Bootstrap 中与本地 Tool Pool 合并，复用同一个 Permission Pipeline、
  PreToolUse/PostToolUse Hooks、Tool 超时和输出截断边界。
- 自动化使用真实本地 stdio Mock MCP Server 完成握手、`tools/list`、`tools/call` 和
  `isError` 验证；默认测试不访问公网或真实模型。
- CLI 提供 `/skills` 和 `/mcp`，通过 Agent Server 分别查看当前用户可见 Skill 与
  进程实际发现的 MCP Server、Tool 和安全属性。
- CLI 提供 `/test tool|skill|mcp <name> <test request>`：Tool/MCP 使用
  `required_tool` 强制首个模型调用，Skill 强制调用 `load_skill` 并在测试指令中固定
  Skill 名称；破坏性 MCP Tool 仍必须经过 Permission interrupt。

- 连接 MCP Server。
- 发现并规范化工具名称。
- 组装 MCP Tool 与本地 Tool Pool。
- 调用 Tool 并转为统一 ToolResult。
- MCP Tool 进入同一个 Permission Pipeline 和 Hooks。
- 连接失败时隔离单个 Server，不破坏本地工具。

#### M5-3 Knowledge Assistant 单 Agent 闭环

实现状态：**自动化与 CLI 手工验收均通过（2026-08-27）**。

第一阶段最终业务场景：

```text
用户要求分析本地文件
  ↓
TodoWrite 制定步骤
  ↓
Task System 创建分析与报告任务
  ↓
file_reader 读取授权文件
  ↓
calculator 执行必要计算
  ↓
report_writer 请求写入审批
  ↓
保存报告 Artifact
  ↓
complete_task 保存任务结果
```

这个场景不能调用 RAG、Subagent 或 Agent Teams。

实现说明：

- Knowledge Assistant System Prompt 已定义文件分析报告闭环：Todo 规划、持久化 Task
  创建与认领、相关 Skill 加载、授权文件读取、必要计算、报告写入、Task 完成/失败和
  Todo 收尾；只有用户明确要求时才创建独立后续 Task。
- `AGENT_AGENT_LOOP__MAX_ITERATIONS` 提供有界迭代配置，默认 16，避免包含 Skill、
  Permission interrupt 和 Task 收尾的合法长闭环被原八轮默认值提前截断。
- Tool Hook 日志已携带 `thread_id` 和 `run_id`，能够把审批前后恢复的工具调用关联到
  同一个 Conversation Run，同时不记录工具参数和完整输出。
- `agent serve` 会在启动 Uvicorn 前按 `AGENT_LOGGING__LEVEL/FORMAT` 初始化标准
  `logging` 与 `structlog`；Uvicorn、Tool Hook 和 MCP 日志进入同一个 Server 输出流。
  Console/JSON 格式只输出白名单关联字段，不序列化 Tool 输入、结果正文或密钥。
- 2026-08-27 CLI 手工复验确认：Tool 调用开始/结束日志能在 Agent Server 终端成对显示，
  且包含 Conversation Run 关联字段。
- Graph 通过不对模型公开的 `ToolExecutionContext` 向上下文感知 Tool 传递可信
  `thread_id/metadata`；Task 创建工具据此自动保存 `conversation_id/run_id`，主任务和
  同一次闭环创建的后续任务拥有相同来源。
- 确定性 FakeModel E2E 已验证 TodoWrite、Task 创建/认领、双文件读取、Calculator、
  ReportWriter 覆盖审批、Checkpoint 恢复、Task 完成、后续 Task 创建和 Todo 完成；
  报告真实写入用户 Artifact 目录，全部 ToolUse/ToolResult 正确配对。
- Knowledge Assistant Profile 未装配 `document_search`、Subagent 或 Agent Teams，闭环
  不依赖 RAG、Redis、后台 Worker、外部 MCP Server 或真实模型。
- CLI 手工验收已完成文件分析、TodoWrite 规划、Task 创建/认领、双文件读取、必要计算、
  ReportWriter 权限分支、Artifact 保存、Task 持久化与来源核验，以及 Tool Hook Server 日志复验。

### 8.4 自动化验证

#### Task System 测试

- Task ID 唯一。
- 状态转换只能按照允许路径进行。
- 依赖未完成时认领失败。
- 多个执行者并发认领时只有一个成功。
- 重启后 Task 状态和结果引用仍存在。

#### MCP 测试

使用 MockMCPServer 验证：

- Server 连接与工具发现。
- 名称冲突时的命名规范化。
- readOnly 工具自动执行。
- destructive 工具触发 Permission。
- Server 超时后本地工具仍可使用。
- MCP 返回错误能转换为标准 ToolResult。

#### 单 Agent E2E

使用 FakeModel 跑完整 CLI 场景，验证调用顺序：

```text
TodoWrite
  → create_task
  → file_reader
  → calculator（可选）
  → report_writer
  → complete_task
```

检查：

- 报告文件真实存在于 Artifact 范围内。
- Task 状态为完成。
- 对话中存在正确的 ToolUse/ToolResult 配对。
- 审批记录、日志和 Trace 可以关联到同一个 `thread_id`。
- 没有调用 `document_search`、Subagent 或 Agent Teams。

### 8.5 手工验证

准备两个本地文件，例如产品说明和需求清单，然后输入：

> 阅读这两个文件，比较主要差异，计算条目数量，生成一份 Markdown 报告，并记录一项后续检查任务。

确认：

- Agent 先规划再执行。
- 文件读取符合授权范围。
- 写报告时按照 PermissionRule 决定是否审批。
- 报告保存成功并返回 Artifact 路径。
- Task System 中存在完成任务和后续任务。

### 8.6 M5 验收标准

- Task System 和 MCP Tools 均有独立自动化测试。
- Knowledge Assistant 完成无 RAG、无 Agent Teams 的端到端业务闭环。
- 所有副作用操作都经过 Permission，并具有明确的结果记录。
- CLI 中断后可以通过 Checkpoint 恢复。
- 默认测试不需要真实模型、Redis、后台 Worker 或外部 MCP Server。
- 第一阶段所有质量门禁通过。

---

## 9. 第一阶段最终验收矩阵

| 能力 | 对应文件 | 自动验证 | 手工验证 | 通过标准 |
| --- | --- | --- | --- | --- |
| AgentProfile | `harness/profile.py` | Profile/Bootstrap 测试 | 切换业务配置 | 业务不直接创建模型或 Loop |
| Agent Loop | `agent_loop.py`、`graph.py` | FakeModel 循环测试 | 普通问答和工具问答 | 正确结束且无无限循环 |
| Tool Use | `tool_use.py` | ToolUse/ToolResult 配对 | calculator/file_reader | 工具只执行一次 |
| Permission | `permissions.py` | 四种决策测试 | 报告覆盖审批 | ALLOW/DENY/ASK/PASSTHROUGH 正确 |
| Hooks | `hooks.py` | 顺序和阻断测试 | 日志与 Stop 统计 | 不侵入 Agent Loop |
| Conversation | `harness/conversation.py` | 多轮、隔离、权限恢复 | Alice/Bob 同时对话 | 相同 Thread 连续、不同用户隔离 |
| 多用户 CLI/API | `entrypoints/cli.py`、`api.py` | ASGI 并发和所有权测试 | 两个终端同时运行 | 不同 Conversation 并发，同一 Conversation 冲突 |
| User Runtime | `entrypoints/bootstrap.py` | Runtime 复用和 Artifact 路径测试 | 检查用户产物目录 | 每用户复用 Loop、全局共享 Gateway |
| TodoWrite | `todo_write.py` | 状态转换测试 | 查看执行计划 | 最多一个 in_progress |
| Skill Loading | `skill_loading.py` | 延迟加载测试 | 调用测试 Skill | 未调用时不加载正文 |
| Context Compact | `context_compact.py` | 固定长会话测试 | 连续多轮对话 | 目标和工具配对不丢失 |
| Memory | `memory.py` | 跨会话和隔离测试 | 新会话恢复偏好 | 相关记忆按需加载 |
| Error Recovery | `error_recovery.py` | 故障注入 | 错误模型配置 | 有上限、可恢复或安全终止 |
| Checkpoint | `services/checkpoint.py` | 重启恢复测试 | 审批时退出后恢复 | 正常恢复不重复已完成 Graph 节点 |
| Task System | `task_system.py` | 依赖和并发认领测试 | 查看任务列表 | 状态持久且转换合法 |
| MCP Tools | `services/mcp_tools.py` | Mock Server 测试 | 连接测试 Server | 工具统一进入 Permission Pipeline |
| Knowledge Assistant | `business/knowledge_assistant/` | CLI E2E | 文件分析并生成报告 | 无 RAG/Agent Teams 完成闭环 |

---

## 10. 第一阶段退出条件

只有全部满足以下条件，才能进入第二阶段 RAG：

- M1～M5 的自动化和手工验收全部通过。
- 默认测试套件不访问公网。
- Agent Loop、Tool Use 和 Permission 没有未解决的高风险缺陷。
- Permission interrupt 的正常恢复不会重复执行已经完成的 Graph 节点；进程在外部副作用提交边界崩溃的 Exactly-Once 不属于本阶段保证。
- Knowledge Assistant 能独立完成文件分析和报告生成。
- 至少两个可信 CLI 用户可以同时连接单进程 Agent Server，且 Conversation、Checkpoint、权限和 Artifact 不串线。
- 同一 Conversation 的并发主 Run 被明确拒绝，不依赖未实现的队列。
- `document_search` 尚未注册到第一阶段 Profile。
- `subagent.py` 和 `agent_teams/` 尚未进入运行路径。
- 代码目录与总体架构一致，没有重新引入已删除文件。

满足退出条件后，第二阶段只需要在现有 Profile 中增加 RAG Capability 和 `document_search`，不修改 Agent Loop 或 ConversationService。
