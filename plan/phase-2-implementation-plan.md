# 第二阶段实施计划：RAG 最小闭环

总体能力边界见[总体实施计划 4.2](./implementation-plan.md#42-第二阶段rag)。本阶段只实现
Knowledge Assistant 当前需要的检索闭环，目标是在一个开发日内完成并验收。

阶段状态：**已实现并通过自动化、pgvector 集成及 FastEmbed 冒烟验收（M6）**。

## 1. 阶段目标

让 Knowledge Assistant 能够把 Markdown/TXT 业务资料写入向量库，通过统一
`document_search` Tool 检索当前用户有权访问的片段，并基于真实来源生成引用。

```text
业务资料 + 元数据
        ↓
加载 → 切分 → Embedding → pgvector
                         ↑
用户问题 → document_search → RAG Pipeline
                              ├── 用户范围过滤
                              ├── Top-K 与分数阈值
                              ├── 去重与 Context 预算
                              └── Citation
```

完成后应满足：

- 公共业务知识和当前用户私有知识可以入库、更新和检索。
- `business/` 不直接依赖 Embedding Provider 或 pgvector。
- `harness/` 只定义通用检索流程和数据契约，不包含知识助手业务字段。
- `services/rag/` 完成文档加载、切分、Embedding 和 pgvector 适配。
- 检索结果具有真实 `source/section/chunk_id`，无结果时不伪造引用。
- Alice 不能检索 Bob 的私有知识。
- 关闭 RAG 后仍能运行第一阶段单 Agent 能力。

### 1.1 当天实现边界

今天必须完成：

- Markdown、TXT 加载。
- Markdown frontmatter 和默认技术元数据。
- 标题/段落优先、长度兜底的文本切分。
- 可替换的 Embedding 协议和一个本地 Embedding 实现。
- PostgreSQL + pgvector 持久化和向量 Top-K。
- 公共/用户私有知识过滤。
- `document_search`、Profile、Bootstrap 和 CLI Indexer 装配。
- `[S1]` 形式的可定位引用。
- 固定检索数据集、自动化验证和一组 CLI 手工验收。

今天不实现的内容见第 8 节。

---

## 2. 技术选型

| 范围 | 选型 | 使用方式 |
| --- | --- | --- |
| 通用协议 | Pydantic v2 + Protocol | 定义 Document、Chunk、Query、Hit、Citation 和 Retriever |
| 文本切分 | `langchain-text-splitters` | Markdown 标题优先，递归字符切分兜底 |
| Embedding | FastEmbed + `BAAI/bge-small-zh-v1.5` | 本地执行，避免新增模型 API Key；通过协议注入 |
| 向量存储 | PostgreSQL 16 + pgvector | 本地 Docker Compose 启动，持久化 Chunk 和向量 |
| 适配 | `langchain-postgres` | 只在 `services/rag/` 内使用 |
| CLI | 现有 Typer | 增加 `agent index`，不新增管理服务 |
| 测试 | FakeEmbeddings + InMemoryVectorStore + pytest | 默认测试不下载模型、不启动 Docker、不访问公网 |

配置增加到 `services/config.py` 的 `RAGSettings`：

```text
AGENT_RAG__ENABLED=false
AGENT_RAG__DATABASE_URL=postgresql+psycopg://...
AGENT_RAG__COLLECTION_NAME=knowledge_assistant
AGENT_RAG__EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
AGENT_RAG__TOP_K=5
AGENT_RAG__SCORE_THRESHOLD=0.35
AGENT_RAG__MAX_CONTEXT_CHARACTERS=12000
```

API Key、数据库地址和模型路径只从配置读取，不写入代码或日志。

---

## 3. 当天实施顺序

```text
M6-0 环境预检（约 30 分钟）
  ↓
M6-1 契约与配置（约 1 小时）
  ↓
M6-2 入库与向量存储（约 2 小时）
  ↓
M6-3 检索、引用与业务装配（约 2 小时）
  ↓
M6-4 评测与验收（约 1～2 小时）
```

后一个任务只依赖前一个任务公开的接口，不允许 `document_search` 绕过 Pipeline 直接查询数据库。

M6-0 先完成依赖安装、FastEmbed 模型首次下载和 pgvector 容器启动；这是当天计划的关键路径。
如果外部下载或 Docker 不可用，可以继续完成 FakeEmbeddings/InMemoryVectorStore 的代码和测试，
但不能把第二阶段标记为验收通过。

---

## 4. M6-1：RAG 契约与配置

### 4.1 涉及文件

```text
src/harness/capabilities/rag/
├── contracts.py
├── pipeline.py
└── citations.py

src/services/config.py
```

### 4.2 核心技术点

`contracts.py` 定义不可变通用模型：

- `SourceDocument`：原始文档、来源和通用元数据。
- `DocumentChunk`：`document_id/chunk_id/text/metadata`。
- `RetrievalQuery`：查询文本、Top-K、分数阈值和可信 Scope；不允许模型填写 `user_id`。
- `RetrievalHit`：Chunk、相关度分数和排名。
- `Citation`：引用 ID、来源、Section 和 Chunk ID。
- `EmbeddingProvider`、`VectorStore` 和 `Retriever` Protocol。

`pipeline.py` 只负责：

1. 校验查询和可信用户范围。
2. 调用 Retriever。
3. 按 `chunk_id` 去重并稳定排序。
4. 应用 Top-K、最低分数和 Context 字符预算。
5. 生成确定性的 `[S1]`、`[S2]` 引用。

`citations.py` 校验每个引用都对应本次返回的真实 Chunk；无命中时返回空结果和明确提示。

### 4.3 验证方案

- FakeRetriever 返回乱序/重复结果时，Pipeline 输出顺序稳定且正确去重。
- 超过预算的低排名 Chunk 被裁剪，高排名来源保留。
- 空检索返回 `matches=[]`，不生成 Citation。
- 模型输入无法覆盖 Runtime 注入的 `user_id`。
- Ruff、Pyright 和相关单元测试通过。

---

## 5. M6-2：文档入库与 pgvector

### 5.1 涉及文件

```text
src/services/rag/
├── embeddings.py
├── splitter.py
├── ingestion.py
└── vector_store.py

src/entrypoints/indexer.py
compose.rag.yaml
```

### 5.2 元数据边界

业务文档可通过 Markdown frontmatter 提供：

```yaml
title: 产品使用说明
category: product_manual
tags: [product, guide]
version: "2026-08"
```

入库服务自动补充：

- `document_id`：知识库、Scope、Owner 和相对路径的稳定摘要。
- `chunk_id`：Document、内容版本和 Chunk 序号的稳定摘要。
- `knowledge_base_id`、`source`、`section`、`content_hash`。
- `scope=public|user`、`user_id`。
- `embedding_model`、`embedding_dimension`、`indexed_at`。

业务负责 `category/tags/version` 的含义；Services 只保存和过滤，不解释其业务语义。

### 5.3 入库流程

```text
扫描 .md/.txt
  ↓
读取正文与 frontmatter
  ↓
路径和 Scope 校验
  ↓
按标题、段落切分；超长段落递归切分并保留 overlap
  ↓
批量 Embedding
  ↓
同一 document_id 删除旧 Chunk，再写入新版本
  ↓
输出 indexed/skipped/deleted/failed 统计
```

增量规则：`content_hash + embedding_model + splitter_version` 未变化时跳过；任一项变化时替换该
Document 的全部 Chunk，避免旧版本继续命中。当天只提供同步 CLI Indexer，不引入 Worker。

CLI 入口：

```bash
.venv/bin/agent index --source src/business/knowledge_assistant/knowledge --scope public
.venv/bin/agent index --source users/alice/knowledge --scope user --user alice
.venv/bin/agent index --source src/business/knowledge_assistant/knowledge --scope public --rebuild
```

### 5.4 验证方案

- Loader 正确处理 Markdown frontmatter、TXT 默认元数据和 UTF-8。
- Splitter 不产生空 Chunk，Section、顺序和 overlap 可复现。
- FakeEmbeddings 批量调用次数和向量维度正确。
- 相同文件重复入库被跳过；修改后旧 Chunk 被替换。
- Embedding 模型或维度变化时拒绝混用旧 Collection。
- pgvector 集成测试使用独立测试 Collection，并在结束后清理。

---

## 6. M6-3：检索、引用与 Knowledge Assistant 装配

### 6.1 涉及文件

```text
src/harness/capabilities/rag/pipeline.py
src/harness/capabilities/rag/citations.py
src/business/knowledge_assistant/tools/document_search.py
src/business/knowledge_assistant/profile.py
src/business/knowledge_assistant/system_prompt.py
src/business/knowledge_assistant/permission_rules.py
src/entrypoints/bootstrap.py
src/entrypoints/cli.py
```

### 6.2 查询流程

```text
document_search(query, category?, tags?, top_k?)
  ↓
Bootstrap 为当前 UserRuntime 绑定可信 user_id Scope
  ↓
分别检索 public 和 user_id 私有 Scope
  ↓
合并、去重、分数过滤、Context 预算
  ↓
返回 snippets + [S1]/[S2] citations
  ↓
模型仅依据命中内容回答并保留引用 ID
```

核心约束：

- `document_search` 是只读 Tool，可以自动允许，但必须经过现有 Permission Pipeline 和 Hooks。
- Tool 参数不暴露 `user_id/tenant_id/collection/database_url`。
- Runtime 创建用户专属 Retriever，不能依赖 Prompt 提醒实现隔离。
- 检索文本属于不可信数据，ToolResult 明确包裹为参考资料，不能覆盖 System Prompt。
- ToolResult 不返回数据库连接信息、绝对路径或未经授权的元数据。
- RAG 关闭时 Profile 不注册该 Tool，第一阶段能力继续工作；RAG 已开启但数据库不可用时
  Bootstrap 明确启动失败，不能静默退化成一个看似可用但无法检索的 Tool。
- 检索异常返回结构化 Tool 错误，不触发模型网关降级，也不重试有问题的查询。

### 6.3 引用格式

ToolResult 使用稳定结构：

```text
[S1] source=product-guide.md section=退款规则 chunk_id=...
退款申请必须在订单完成后七天内提交。

[S2] source=service-policy.md section=特殊商品 chunk_id=...
虚拟商品不适用无理由退款。
```

System Prompt 要求：

- 只引用当前 ToolResult 中存在的 ID。
- 无命中时说明知识库没有相关资料。
- 区分知识库事实和模型自己的解释。

### 6.4 验证方案

- `document_search` 只能通过 RAG Pipeline 调用 VectorStore。
- Alice 查询结果包含公共知识和 Alice 私有知识，不包含 Bob 私有知识。
- ToolUse/ToolResult ID 配对，Hook 日志包含 `thread_id/run_id/tool_name`，不记录查询正文。
- FakeModel E2E 验证模型调用 `document_search` 后输出真实 `[S1]` 引用。
- `AGENT_RAG__ENABLED=false` 时 Profile 中没有 `document_search`，普通问答和第一阶段测试不受影响。

---

## 7. M6-4：检索评测与最终验收

### 7.1 固定评测集

在 `tests/fixtures/rag/evaluation.jsonl` 准备 5～10 条小型数据：

```json
{"query":"退款可以在几天内申请？","expected_sources":["product-guide.md"]}
```

当天自动计算：

- `Recall@K`：期望来源是否进入前 K。
- `MRR`：第一个正确来源的排名。
- Citation validity：引用 ID 是否指向本次真实命中。
- Isolation failures：跨用户错误命中数，必须为 0。

首版不把模型生成答案的主观质量作为 CI 硬门禁；回答正确性通过一条 CLI 冒烟场景确认。

### 7.2 CLI 手工验收

1. 启动 pgvector 并执行公共资料、Alice 私有资料和 Bob 私有资料入库。
2. 启动 Agent Server，Alice 进入 Chat。
3. 使用 `/test tool document_search ...` 确认 Tool 返回片段和真实引用。
4. 普通提问要求 Agent 基于资料回答，确认回答包含 `[S1]`。
5. Alice 查询 Bob 私有资料的唯一测试代号，确认无结果。
6. 关闭 `AGENT_RAG__ENABLED` 并重启，确认普通聊天、Calculator 和 FileReader 仍可使用。

### 7.3 第二阶段验收矩阵

| 能力 | 自动验证 | 手工验证 | 通过标准 |
| --- | --- | --- | --- |
| 文档加载与切分 | Markdown/TXT、边界和稳定性测试 | 检查入库统计 | Chunk 非空且来源可定位 |
| 增量索引 | 重复、更新和旧版本测试 | 修改文档后重建 | 旧 Chunk 不再命中 |
| Embedding | Fake 批处理和维度测试 | 本地模型完成向量化 | 不混用不同维度 |
| pgvector | Collection 集成测试 | 重启后继续检索 | 数据持久且 Top-K 正确 |
| Pipeline | 去重、阈值和预算测试 | 强制调用 Tool | 结果稳定且有界 |
| 用户隔离 | Alice/Bob 过滤测试 | 私有代号交叉查询 | 跨用户命中为 0 |
| Citation | ID 与 Chunk 对应测试 | 回答检查 `[S1]` | 引用可定位且不伪造 |
| 降级 | RAG 关闭/不可用测试 | 关闭 RAG 后聊天 | 第一阶段能力可继续运行 |

### 7.4 统一质量门禁

```bash
.venv/bin/ruff check src tests
.venv/bin/pyright src
.venv/bin/pytest -m "not live_model and not rag_integration"
.venv/bin/pytest -m rag_integration
```

默认测试使用 FakeEmbeddings 和 InMemoryVectorStore，不访问公网、不下载模型、不依赖 PostgreSQL。

---

## 8. 今天明确不做

以下内容不阻塞第二阶段最小闭环，后续按检索质量和业务需求增加：

- PDF、Office、网页抓取、OCR 和多模态文档。
- LLM Query Rewrite、多轮问题改写和 HyDE。
- BM25/全文检索混合召回。
- Cross-Encoder 或外部 Reranker。
- 精确模型 tokenizer；首版沿用可配置字符预算。
- 自动摘要 Chunk、实体抽取和知识图谱。
- 文件监听、后台增量任务、定时重建和管理后台。
- 生产身份认证、租户管理、行级安全策略和多 Worker。
- 大规模离线评测、答案忠实度模型评审和线上 A/B 测试。

预留 `Reranker`、metadata filters 和 Provider Protocol，但不实现无调用方的复杂逻辑。

---

## 9. 第二阶段退出条件

只有以下条件全部满足，才进入第三阶段 Agent Teams：

- Markdown/TXT 可以通过 CLI 完成可重复入库和更新。
- Knowledge Assistant 可以调用 `document_search` 并生成可定位引用。
- 公共知识共享、用户私有知识隔离，跨用户命中数为 0。
- 无结果时明确说明，不构造虚假 Citation。
- RAG Pipeline 不依赖知识助手业务概念，业务 Tool 不依赖 pgvector。
- 关闭 RAG 后第一阶段完整测试继续通过。
- 默认自动化测试不依赖真实模型、Embedding 下载或外部数据库。
- pgvector 集成测试、固定检索评测和至少一条 CLI 真实闭环通过。
- Ruff、Pyright 和 Pytest 质量门禁通过。
