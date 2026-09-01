# 第三阶段实施计划：Multi-Agent

总体能力边界见[总体实施计划 4.3](./implementation-plan.md#43-第三阶段multi-agent)。本阶段在第一阶段
单 Agent 和第二阶段 RAG 闭环之上，引入 Lead Agent、Subagent/Teammate 和 Reviewer，验证
Knowledge Assistant 可以把复杂研究任务拆分、并行执行、审核后再汇总。

阶段状态：**M7-0～M7-2、M8-1～M8-2 核心实现已完成，M8-3 综合验收中**。当前已完成
`harness/capabilities/agent_teams/` 的子 Agent 契约、隔离运行时、预算、委派协调、
进程内 MessageBus、Team Protocols、任务级重试和审核轮次限制，并接入
`business/knowledge_assistant/agent_teams/` 的 Lead/Researcher/Analyst/Reviewer 与 Bootstrap。

## 1. 阶段目标

交付一个只使用进程内资源的 Multi-Agent 最小闭环：

```text
用户问题
    ↓
Lead Agent 制定计划和拆分任务
    ├── Researcher：检索资料并整理证据
    ├── Analyst：比较、计算并形成结论
    └── Reviewer：检查结论、计算和引用
    ↓
Lead Agent 根据审核结果汇总
    ↓
返回带来源的最终回答或进入有限次数重做
```

完成后应满足：

- Lead 能根据任务类型选择合适的 Teammate，并为每个任务生成明确的目标和输出 Schema。
- Lead 能为不同研究任务创建拥有不同 Tool Allowlist 的 Researcher 实例，并返回结构化证据。
- Analyst 只处理 Lead 传入的必要证据，不直接读取完整主会话。
- Reviewer 能校验事实、计算、来源和结论边界，并在不合格时给出可执行的修改意见。
- 无依赖的 Researcher/Analyst 任务可以并行执行；有依赖的任务按协议等待前置结果。
- 每个 Teammate 拥有独立的子状态、上下文、工具白名单和预算，但继承可信用户的授权范围。
- Teammate 的失败、超时或拒绝不会破坏 Lead 的主状态；Lead 可以降级、重试或向用户说明失败。
- `ASK` 权限请求可以从 Teammate 冒泡到 Lead，并在原任务中恢复，不重复执行已完成节点。
- 现有单 Agent、RAG、CLI、MCP 和用户隔离测试继续通过；关闭 Multi-Agent 时不影响已有能力。

### 1.1 本阶段明确不实现

- Redis 或其他跨进程 MessageBus 持久化。
- 独立 Worker、后台任务、Cron、长任务队列和多 Worker 部署。
- 正式身份认证、租户管理和生产级行级安全。
- 完全自治的 Agent 网络、无限递归委派和 Teammate 自由创建 Teammate。
- 跨进程恢复、崩溃边界下的副作用幂等；这些属于第四阶段。
- Multi-Agent 的前端、Token Streaming、远程取消和跨设备交互。
- 让 Teammate 直接写报告、修改业务数据或绕过 Lead 执行高风险操作。

### 1.2 实施原则

1. `harness/capabilities/agent_teams/` 只定义通用协议、调度和隔离机制，不出现面试、文档或业务字段。
2. `business/knowledge_assistant/agent_teams/` 只定义角色 Prompt、工具白名单和输出 Schema，不实现并发调度。
3. Lead 与 Teammate 之间只传结构化任务和结果，不通过共享可变全局状态通信。
4. 主会话的用户身份、Scope、Permission 和搜索模式由 Runtime 注入，不能由模型或子 Agent 改写。
5. 检索内容、MCP 结果和 Teammate 输出均视为不可信资料，不能覆盖 System Prompt 或权限规则。
6. 所有预算、最大深度、最大重做次数和超时都由运行时硬限制，不能只写在 Prompt 中。

## 2. 前置条件与完成定义

第三阶段开始前必须确认：

- 第一阶段退出条件全部满足，Conversation、Checkpoint、Permission、Hooks 和 Model Gateway 可用。
- 第二阶段退出条件全部满足，RAG 检索、引用、增量索引和用户隔离可用。
- `document_search`、`document_catalog`、Calculator 和现有 MCP Tool 均可通过统一 Tool/Permission Pipeline 调用。
- FakeModel、FakeRetriever、FakeTool 和 InMemory Store 可以独立驱动测试，不要求真实模型或公网。
- 现有 `SearchMode` 能作为 Team 上下文的一部分传递给 Researcher；`rag`、`web`、`hybrid` 的限制不能因委派而失效。

本阶段完成定义：

1. M7 和 M8 的自动化测试、集成测试和 CLI 手工场景全部通过。
2. 默认测试不依赖真实模型、公网、Redis 或后台 Worker。
3. Lead、Researcher、Analyst、Reviewer 的输入输出都经过 Pydantic Schema 校验。
4. 每次委派都有可关联的 `team_run_id`、`task_id`、`parent_task_id` 和角色信息。
5. 至少完成一次“检索—分析—审核—汇总”的真实 RAG 闭环，并保留真实 Citation。
6. 至少覆盖一次 Teammate 失败、超时、权限请求和 Reviewer 退回重做场景。
7. Ruff、Pyright 和 Pytest 质量门禁通过。

## 3. 实施顺序

第三阶段拆成两个正式里程碑和一个阶段验收：

```text
M7-0 前置检查与 Team 契约
  ↓
M7-1 Subagent 隔离运行时
  ↓
M7-2 Lead + Researcher 委派闭环
  ↓
M8-1 进程内 MessageBus 与 Team Protocols
  ↓
M8-2 Analyst + Reviewer 并行、审核和有限重做
  ↓
M8-3 权限、观测、错误恢复和综合验收
```

| 任务 | 主要内容 | 完成标志 |
| --- | --- | --- |
| M7-0 | 前置条件、接口边界和 Fake 实现 | 契约评审通过，未改变单 Agent 默认路径 |
| M7-1 | Subagent 创建、上下文裁剪、工具白名单和预算 | 子 Agent 无法读取无关主会话 |
| M7-2 | Lead、Researcher、Agent-as-Tool 委派 | Lead 能完成一次检索任务并接收结构化结果 |
| M8-1 | MessageBus、Task/Result/Permission/Shutdown 协议 | 请求响应、关联 ID 和关闭流程正确 |
| M8-2 | Analyst、Reviewer、并行和重做 | 审核失败可有限次重做，结果不污染主状态 |
| M8-3 | 安全、追踪、异常、CLI 和综合验收 | 退出条件全部满足 |

后一个任务只依赖前一个任务公开的协议；业务角色不能绕过 `team.py` 直接创建线程、任务或共享状态。

### 3.1 第三阶段 TODO（必做项）

- [ ] 实现 Multi-Agent 子任务级重试，不重跑整个 Team。
- [ ] 按错误类型区分自动重试、直接失败、等待权限和降级处理。
- [ ] 为每个子任务设置最大尝试次数、总耗时和 Token 预算，并记录 `attempt`、`last_error` 和状态变化。
- [ ] 使用指数退避和随机抖动，避免多个失败任务同时重试造成请求尖峰。
- [ ] 保留已成功的兄弟任务结果，只重新执行失败、超时或被 Reviewer 退回的任务。
- [ ] 为有副作用的 Tool 增加 `idempotency_key` 或等价去重机制，确保重试不重复写入。
- [ ] 处理任务依赖：失败任务的下游任务进入 `BLOCKED` 或按策略使用部分结果继续。
- [ ] 覆盖重试成功、超过上限、不可重试错误、超时和重复投递测试。

## 4. M7：Subagent 与 Lead/Researcher 最小闭环

### 4.1 涉及文件

```text
src/harness/capabilities/subagent.py
src/harness/capabilities/agent_teams/
├── __init__.py
├── contracts.py
└── team.py

src/business/knowledge_assistant/agent_teams/
├── __init__.py
├── lead.py
└── researcher.py

src/business/knowledge_assistant/profile.py
src/business/knowledge_assistant/system_prompt.py
src/entrypoints/bootstrap.py
```

测试文件：

```text
tests/unit/capabilities/test_subagent.py
tests/unit/capabilities/agent_teams/test_contracts.py
tests/unit/capabilities/agent_teams/test_team.py
tests/unit/business/test_knowledge_assistant_team.py
tests/integration/test_agent_team_research.py
```

### 4.2 通用契约

在 `contracts.py` 中定义不可变、可序列化的模型：

- `AgentRole`：角色名、描述和允许的能力。
- `TeamTask`：`task_id`、`parent_task_id`、角色、目标、结构化输入、上下文、工具白名单和预算。
- `TaskStatus`：`pending/running/succeeded/failed/timeout/cancelled/rejected`。
- `TeamTaskResult`：任务状态、结构化输出、引用、错误、耗时和使用量摘要。
- `DelegationBudget`：最大委派深度、任务数、模型轮数、Token、总耗时和重做次数。
- `SubagentContext`：经过筛选的任务上下文、可信身份 Scope、搜索模式和允许的工具。
- `TeamRun`：一次 Team 执行的 `team_run_id`、父级 `run_id`、状态和预算快照。

所有跨角色结果必须包含：

```text
task_id
role
status
summary
evidence[]
citation_ids[]
warnings[]
usage
```

业务 Schema 可以增加 `company`、`interview_round` 等字段，但不能把这些字段放进 Harness 契约。
Schema 校验失败必须返回结构化失败结果，不能让 Lead 解析任意自然语言作为成功结果。

### 4.3 Subagent 隔离运行时

`subagent.py` 负责把一个 `TeamTask` 映射为独立的 Agent Loop 或 LangGraph Subgraph：

```text
Lead State
  ↓ 仅复制任务所需字段
SubagentContext
  ↓
独立 thread_id / 子状态 / Prompt / Tool 白名单
  ↓
Subagent Agent Loop
  ↓
TeamTaskResult
  ↓ 仅回传结果给 Lead
```

必须实现：

- 子 Agent 使用独立 `thread_id`，不能直接写入 Lead 的消息列表或 Checkpoint。
- Context 只包含任务目标、必要证据、输出 Schema、用户授权 Scope、搜索模式和预算。
- 工具列表由 Lead/Runtime 按任务明确传入；每个 Subagent 只获得完成当前任务所需的最小 Tool Allowlist，不能使用 `report_writer` 等无关工具。
- 子 Agent 继承可信 `user_id` 和权限上下文，但工具参数中不暴露可伪造的所有权字段。
- 子 Agent 不能创建新的子 Agent；第一版最大委派深度为 1。
- 子 Agent 结束时释放临时资源并返回结构化状态，异常必须被 `team.py` 转换为失败结果。
- 子 Agent 的上下文长度和 ToolResult 数量受独立预算控制，不得耗尽主会话预算。

### 4.4 Lead 与 Researcher

`lead.py` 只负责业务层的计划和汇总约束：

- 识别问题是简单问答还是需要 Team 的研究/比较/分析任务。
- 把任务拆分成最小的、可验证的 Researcher 任务。
- 为任务指定输入、输出 Schema、检索模式和允许工具。
- 合并成功、失败和部分成功结果，并把不确定性传递给最终回答。
- 不在没有 Reviewer 结果时声称“已审核”或把推断写成事实。

`researcher.py` 定义研究角色的通用行为，但具体实例按任务绑定不同工具：

| Researcher 实例 | 允许工具 | 适用任务 |
| --- | --- | --- |
| `CatalogResearcher` | `document_catalog` | 数量、枚举、文档清单和精确元数据筛选 |
| `RAGResearcher` | `document_search` | 检索已授权知识库并保留 `[S1]` 等真实引用 |
| `WebResearcher` | 已发现的 Web Search MCP Tool | 新闻、价格、行情和其他最新公开信息 |

默认不把三类工具全部放给同一个 Researcher。`hybrid` 模式通常拆成 `RAGResearcher` 和
`WebResearcher` 两个并行任务，Lead 再合并两类证据；只有确有必要时才创建同时拥有两类搜索工具的专用实例。
没有对应 Web Tool 时，`WebResearcher` 返回明确不可用状态，不自行降级到 RAG。

Team 必须继承主会话的 `search_mode`：`rag` 禁止 Web Tool，`web` 禁止本地 RAG，`hybrid` 才允许同时使用。
Researcher 不能通过新建子任务、修改 Prompt 或扩大 Tool Allowlist 绕过这一限制。

### 4.5 M7 验收

```text
/search-mode rag
请从我的面试记录中列出字节相关的面试文档，并给出每份记录的引用。
```

验收：Lead 选择 Researcher；结果来自当前用户可访问的 RAG；引用真实可定位；日志包含
`team_run_id`、`task_id`、角色和状态；关闭 Multi-Agent Capability 后原有 `document_search` 仍可用。

## 5. M8：MessageBus、Analyst、Reviewer 与审核闭环

### 5.1 涉及文件

```text
src/harness/capabilities/agent_teams/
├── message_bus.py
└── team_protocols.py

src/business/knowledge_assistant/agent_teams/
├── analyst.py
└── reviewer.py

src/business/knowledge_assistant/profile.py
src/harness/hooks.py
src/harness/permissions.py
src/harness/error_recovery.py
src/harness/logging.py
```

### 5.2 进程内 MessageBus

`message_bus.py` 只实现进程内异步通信，但接口要为第四阶段持久化实现保留替换点：

- `send(message)`：向指定角色或任务发送一条消息。
- `publish(event)`：发布任务状态、结果和错误事件。
- `subscribe(topic, handler)`：订阅并按顺序处理事件。
- `close(team_run_id)`：拒绝新消息并等待当前消息完成。
- `drain(team_run_id)`：测试和关闭时等待已发布消息处理完毕。

消息至少包含：

```text
message_id
team_run_id
parent_run_id
task_id
parent_task_id
sender
recipient
kind
correlation_id
payload
created_at
```

本阶段在内存中完成顺序投递、并发任务隔离和关闭；不承诺进程崩溃后恢复。重复消息在同一个
`team_run_id + message_id` 范围内应被安全忽略，为第四阶段至少一次投递做准备。

### 5.3 Team Protocols

在 `team_protocols.py` 中定义以下协议，不允许角色之间用非结构化字符串约定状态：

| 协议 | 发起方 | 结果 | 用途 |
| --- | --- | --- | --- |
| `TASK_REQUEST` | Lead | `TASK_ACCEPTED` 或 `TASK_REJECTED` | 委派任务并校验角色、预算和工具 |
| `TASK_RESULT` | Teammate | `RESULT_ACK` | 返回结构化结果和证据 |
| `TASK_FAILED` | Teammate/Runtime | `RETRY` 或 `DEGRADED` | 统一失败和降级 |
| `REVIEW_REQUEST` | Lead | `REVIEW_RESULT` | 请求 Reviewer 审核 |
| `REVISION_REQUEST` | Reviewer | `TASK_RESULT` | 在预算内退回修改 |
| `PERMISSION_REQUEST` | Teammate | `PERMISSION_RESPONSE` | 将 ASK 冒泡到 Lead/用户 |
| `SHUTDOWN` | Lead/Runtime | `SHUTDOWN_ACK` | 终止剩余任务并释放资源 |

协议处理必须校验发送者、接收者、任务归属、Schema、状态转换和关联 ID。未知消息、过期任务或不匹配
的 `correlation_id` 进入结构化错误路径，不得直接交给模型解释。

### 5.4 Analyst

`analyst.py` 只接受 Researcher 已返回的证据和任务参数：

- 对多个来源做比较、归类和计算。
- 需要数字运算时调用受限 Calculator，而不是心算关键结果。
- 为每个结论保留 `citation_ids` 或明确标记为推断。
- 发现证据冲突时列出冲突，不擅自选择看似合理的一条。
- 不直接重新检索无关资料，不访问主会话历史，不修改业务数据。

Analyst 输出建议包含：

```text
findings[]
calculations[]
conclusions[]
supporting_citations[]
uncertainties[]
```

### 5.5 Reviewer

`reviewer.py` 作为独立审核角色，不能默认信任 Lead、Researcher 或 Analyst 的文字结论。至少检查：

- 每个 Citation 是否存在且指向本次真实检索结果。
- 关键事实是否被证据支持，推断是否明确标注。
- 计算输入、公式、单位和结果是否一致。
- 是否混淆公共知识、用户私有知识、Web 证据和模型推断。
- 是否违反用户问题、搜索模式、权限和输出格式要求。
- 是否存在遗漏的失败任务或未解决的证据冲突。

审核结果为：

```text
approved
needs_revision
rejected
```

`needs_revision` 必须带结构化 `issues[]` 和对应的 `task_id/citation_ids`。单个任务最多重做 2 次；
超过次数后由 Lead 返回部分结果和未解决问题，不允许无限循环。

### 5.6 并行和依赖调度

`team.py` 使用 `asyncio.TaskGroup` 或等价进程内机制执行无依赖任务：

```text
Lead
  ├─ Researcher: 公司/文档目录
  ├─ Researcher: 面试轮次和时间
  └─ Researcher: 相关原始片段
       ↓ 全部完成或部分失败
     Analyst
       ↓
     Reviewer
       ↓
     Lead 汇总
```

规则：

- 同一父任务下的无依赖任务可以并行，但每个任务使用独立子状态和预算。
- Analyst 必须等待其声明的 Researcher 依赖完成；部分失败时根据 Schema 决定继续还是降级。
- Reviewer 必须看到最终待审的结构化结果和来源，不直接读取所有中间状态。
- 取消或超时父任务时，向所有子任务发送 `SHUTDOWN`，并等待可控时间后标记剩余任务取消。
- 任务完成顺序不影响最终结果排序；输出按稳定的计划顺序和 `task_id` 排序。

### 5.7 M8 验收

- 两个独立 Researcher 使用 FakeModel/FakeRetriever 并行执行，验证总耗时接近最长任务而非两者相加。
- 制造一个 Researcher 失败，确认其他任务和 Lead 仍可完成，最终回答明确标记资料缺口。
- 制造一个可重试的子任务失败，确认只重试该子任务，已成功的兄弟任务不重复执行。
- 确认重试使用退避、受最大尝试次数限制，并记录每次尝试和最终错误。
- 确认有副作用的工具在重试时不会重复写入或产生重复事件。
- 让 Analyst 计算错误，Reviewer 返回 `needs_revision`，第二次结果通过后停止重做。
- 让 Reviewer 连续拒绝，确认达到上限后返回失败原因，不发生无限循环。
- 在 Teammate 请求 `ASK` 时确认用户审批发生在 Lead 层，恢复后只继续原始任务。
- 发送重复、乱序和未知协议消息，确认不会重复执行工具或污染任务状态。

## 6. 权限、上下文与错误处理

### 6.1 权限冒泡

权限链路必须保持单一入口：

```text
Teammate ToolUse
  ↓
现有 Permission Pipeline
  ├─ DENY → 返回 Teammate 失败结果
  ├─ ALLOW → 执行工具
  └─ ASK → PermissionRequest → Lead → 用户
                              ↓
                       PermissionResponse
                              ↓
                         原任务恢复
```

- Teammate 不能自行批准自己的 `ASK`。
- Lead 不能替用户批准高风险操作；用户身份和审批结果由 Runtime 绑定。
- 同一个 `tool_use_id` 只能成功执行一次；本阶段对有副作用工具默认不开放给 Teammate。
- Reviewer 不拥有写入权限；报告写入必须由 Lead 在最终审核和现有权限规则之后执行。
- `rag`、`web`、`hybrid` 的 SearchModePermissionRule 必须在子 Agent 中继续生效。

### 6.2 上下文预算

预算分为 Team、Task 和模型请求三级：

- Team：总 Token、总耗时、最大任务数、最大并发数和最大重做次数。
- Task：单个角色的 Token、Tool 调用次数、上下文字符数和超时时间。
- Model request：现有 Model Gateway 的单次请求超时、重试和降级策略。

预算消耗写入使用量摘要，不把完整提示词、私有正文或向量写入普通日志。Context Compact 只压缩
子 Agent 自己的历史；Lead 只接收结果摘要和必要证据，不把所有中间消息拼回主会话。

### 6.3 错误分类和降级

至少区分：

- `validation_error`：输入或输出 Schema 不合法，不自动无限重试。
- `permission_denied`：权限拒绝，向 Lead 返回原因和可替代路径。
- `tool_error`：工具失败，按现有工具错误策略有限重试。
- `model_error`：模型临时失败，交给 Model Gateway 处理，不重复创建 Team。
- `timeout`：角色超时，取消其子任务并返回部分结果。
- `protocol_error`：消息关联或状态非法，终止当前任务并保留审计信息。
- `budget_exceeded`：预算用尽，停止委派并返回可解释的部分结果。

Lead 的降级顺序：复用已成功结果 → 有限重试失败任务 → 跳过不影响结论的可选任务 → 返回部分结果和缺口。
不能在没有证据时用模型记忆补齐“看起来完整”的答案。

## 7. 可观测性与数据边界

每次 Team 执行至少记录以下结构化字段：

```text
agent_name
team_run_id
parent_run_id
task_id
parent_task_id
role
message_id
event
status
tool_name
duration_ms
token_usage
error_type
retry_count
search_mode
```

日志和追踪不得记录：用户问题全文、私有文档正文、向量、API Key、Authorization Header 或完整 Prompt。
在调试需要查看证据时，只记录受控的 `source/chunk_id/citation_id` 和脱敏摘要。

每次委派应可回答：谁创建了任务、传入了什么类型的上下文、使用了哪些工具、何时完成、消耗多少、
结果是否审核通过、失败是否重试以及最终如何汇总。

## 8. 测试与验收

### 8.1 单元测试

- Team/Task/Result/Message/Protocol Schema 的校验和状态转换。
- Subagent 上下文裁剪、工具白名单、身份继承和预算扣减。
- Lead 的任务拆分、稳定排序和部分失败聚合。
- MessageBus 的投递、订阅、关闭、重复消息和异常隔离。
- Team Protocols 的关联 ID、发送者/接收者和非法状态校验。
- Researcher/Analyst/Reviewer 的业务输出 Schema。
- Citation 合法性、SearchMode 传递和 Permission ASK 冒泡。
- 重做次数、并发数、超时、取消和预算边界。

### 8.2 集成测试

- FakeModel + InMemoryMessageBus 完成 Lead → Researcher → Analyst → Reviewer 闭环。
- 两个独立任务并行执行，验证结果顺序稳定且状态不互相覆盖。
- RAG 集成测试验证 Alice/Bob 隔离在 Teammate 内仍然有效。
- 使用 Mock MCP 验证 `web`、`rag`、`hybrid` 搜索模式不会被子 Agent 绕过。
- `interrupt()` 审批后恢复原 Teammate 任务，已完成 Tool 不重复执行。
- 任一角色失败、超时或输出非法时，Lead 可以安全降级。

### 8.3 端到端 CLI 验收

准备已索引的公共资料和用户私有资料，执行：

```bash
uv run agent serve
uv run agent chat --user alice
```

场景一：检索和统计。

```text
/search-mode rag
请基于我的面试记录，统计各家公司面试次数，区分明确记录和推断，并给出引用。
```

场景二：需要分析和审核。

```text
/search-mode hybrid
请结合我的知识库和最新公开资料，比较这几家机器人公司的业务方向，分别标注内部资料和联网资料来源。
```

场景三：故障和权限。

```text
请分析资料并把结论保存成报告。
```

验收重点：

- 复杂问题会进入 Team；简单问题不会无条件启动所有角色。
- Researcher、Analyst 和 Reviewer 的职责、工具和上下文边界清晰。
- 回答区分 RAG Citation、Web Citation、计算结果和推断。
- Reviewer 不通过时只有限重做，不出现死循环。
- 报告写入仍由现有 Permission 流程确认，Teammate 不绕过审批。

### 8.4 质量门禁

```bash
uv run ruff check src tests
uv run pyright src
uv run pytest -m "not live_model and not rag_integration"
uv run pytest -m rag_integration
```

默认测试使用 FakeModel、FakeRetriever、FakeTool 和 InMemoryMessageBus，不访问公网、不启动 Redis、不下载模型。

## 9. 第三阶段验收矩阵

| 能力 | 自动验证 | 手工验证 | 通过标准 |
| --- | --- | --- | --- |
| Team 契约 | Schema、状态转换和非法消息测试 | 查看任务事件 | 所有任务可追踪且状态合法 |
| Subagent 隔离 | 主会话泄漏、工具白名单、Scope 测试 | 角色职责检查 | 子 Agent 只能看到任务所需内容 |
| Lead 委派 | FakeModel 路由和聚合测试 | 复杂问题触发 Team | 委派角色和任务目标正确 |
| Researcher | RAG/MCP/目录 Tool 集成测试 | 搜索模式手工验收 | 保留真实来源并遵守模式 |
| 并行执行 | TaskGroup 和时序测试 | 多任务研究场景 | 无依赖任务并行且结果稳定 |
| Analyst | Schema、Calculator、冲突证据测试 | 比较/统计问题 | 结论与证据、计算一致 |
| Reviewer | 通过、退回、拒绝和上限测试 | 检查最终回答 | 不合格结果可解释且有限重做 |
| Permission | ASK/ALLOW/DENY 和恢复测试 | 报告保存审批 | 权限不被子 Agent 绕过 |
| 错误恢复 | 失败、超时、取消和预算测试 | 注入故障 | Lead 状态不损坏，降级明确 |
| 可观测性 | 字段和敏感信息断言 | 查看 Team 日志 | 可关联、可审计、不泄露正文和凭据 |
| 兼容性 | 第一、二阶段全量回归 | 关闭 Team Capability | 原有单 Agent/RAG 行为不回归 |

## 10. 阶段退出条件与第四阶段交接

只有以下条件全部满足，才进入第四阶段生产化：

- Lead、Researcher、Analyst、Reviewer 完成一次真实 RAG 研究闭环。
- 无依赖任务可并行，有依赖任务不会提前执行，结果顺序稳定。
- 子 Agent 上下文、工具、用户 Scope 和 SearchMode 隔离测试全部通过。
- Reviewer 能发现虚假引用、无依据结论和计算错误，并在最多两次内触发修正。
- Teammate 的失败、超时、拒绝和 `ASK` 恢复均有自动化覆盖。
- 不重复执行已经完成的 Tool；本阶段所有可写 Tool 均由 Lead 和 Permission 流程控制。
- Team 运行事件包含完整关联字段，敏感信息不进入日志。
- 第一、二阶段回归测试、Ruff、Pyright 和 CLI 手工验收全部通过。

第四阶段继续处理：

- `services/message_bus.py` 的 Redis/持久化实现。
- PostgreSQL Checkpoint、Team Run/Task Store 和进程崩溃恢复。
- Background Tasks、Worker、Cron、远程取消和流式结果。
- 多 Worker 下的任务租约、幂等、重新投递和跨进程并发控制。
- 正式身份认证、租户隔离、审计存储和生产级 API。
