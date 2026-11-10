# 「安心保」保险 AI 智能顾问

在线保险商城 + AI 保险顾问（Agent + RAG）。

## 目录说明

```
insurance-ai-advisor/
├── agent-service/     后端服务（Python + FastAPI）
├── frontend/          前端商城（React + Ant Design + Vite）
└── deploy/            Docker 编排、数据库初始化脚本
```

## 后端快速开始

```bash
cd agent-service

# 1. 建虚拟环境（指向已装好的 Python 3.11）
uv venv --python E:/Python3.11/python.exe

# 2. 按 pyproject.toml 安装依赖
uv sync

# 3. 复制环境变量模板并填入你自己的 Key
copy .env.example .env

# 4. 先确保 PostgreSQL 已建好 insurance_biz / insurance_agent 两个库。
#    本地开发可先启动 deploy 里的 postgres 容器（会自动执行 initdb 建库）：
#    cd deploy && docker compose up -d postgres

# 5. 建表（两个库分别执行）
uv run alembic -c alembic.ini upgrade head
uv run alembic -c alembic.agent.ini upgrade head

# 6. 灌入演示数据（8 个保险产品、38 条条款、1 个测试账号）
uv run python scripts/seed_data.py

# 7. 条款向量化入库（第一次要加载模型，约 2 分钟）
uv run python scripts/ingest_clauses.py

# 8. 启动
uv run uvicorn app.main:app --reload
```

启动后访问 http://127.0.0.1:8000/health 应返回 `{"status":"ok"}`，
访问 http://127.0.0.1:8000/health/deps 可检查两个数据库和 Redis 是否连通，
访问 http://127.0.0.1:8000/docs 可看到自动生成的接口文档。

测试账号：`13800000000` / `demo1234`

## 已实现的接口

| 方法 | 路径 | 说明 | 是否需要登录 |
| --- | --- | --- | --- |
| POST | `/api/v1/auth/register` | 注册，成功后直接返回令牌 | 否 |
| POST | `/api/v1/auth/login` | 登录，返回 JWT | 否 |
| GET | `/api/v1/auth/me` | 获取当前登录用户 | 是 |
| POST | `/api/v1/auth/logout` | 退出登录，把令牌加入 Redis 黑名单 | 是 |
| GET | `/api/v1/products` | 产品列表，支持险种筛选、关键词搜索、分页 | 否 |
| GET | `/api/v1/products/{id}` | 产品详情，含全部条款 | 否 |
| POST | `/api/v1/products/premium` | 保费试算，返回计算过程 | 否 |
| POST | `/api/v1/orders` | 投保下单 | 是 |
| GET | `/api/v1/orders` | 我的订单 | 是 |
| GET | `/api/v1/orders/{order_no}` | 订单详情 | 是 |
| POST | `/api/v1/orders/{order_no}/pay` | 支付并自动出保单 | 是 |
| POST | `/api/v1/orders/{order_no}/cancel` | 取消订单 | 是 |
| GET | `/api/v1/policies` | 我的保单 | 是 |
| GET | `/api/v1/policies/{policy_no}` | 保单详情 | 是 |
| POST | `/api/v1/claims` | 提交理赔申请 | 是 |
| GET | `/api/v1/claims` | 我的理赔记录 | 是 |
| GET | `/api/v1/claims/{claim_no}` | 理赔详情 | 是 |
| POST | `/api/v1/agent/clause-search` | 条款检索（稠密+稀疏+RRF+重排） | 是 |
| POST | `/api/v1/agent/chat` | 与 AI 顾问对话（完整 Agent 流程） | 是 |
| POST | `/api/v1/agent/chat/stream` | 流式对话（SSE，逐字返回） | 是 |
| GET | `/api/v1/conversations` | 我的会话列表 | 是 |
| GET | `/api/v1/conversations/{id}/messages` | 会话的消息记录 | 是 |
| GET | `/health` | 服务健康检查 | 否 |
| GET | `/health/deps` | 数据库与 Redis 连通性检查 | 否 |

带令牌调用的写法：请求头加 `Authorization: Bearer <access_token>`。

完整业务闭环：登录 → 浏览产品 → 保费试算 → 下单 → 支付（自动出保单）→ 申请理赔。

## 条款向量化（RAG 入库）

```bash
# 增量：只处理还没入库的条款
uv run python scripts/ingest_clauses.py

# 全量重建：清空集合和切片表后重来（改了切分参数或换了模型时用）
uv run python scripts/ingest_clauses.py --rebuild
```

改了条款内容需要重新跑一次；只改查询语句或检索策略不需要重新入库。

Milvus 集合同时存两种向量：

- `dense` 1024 维稠密向量，由 Qwen3-Embedding-0.6B 生成，负责语义匹配
- `sparse` 稀疏向量，由 Milvus 的 BM25 函数用 `search_text` 字段自动生成，负责关键词匹配

`search_text` = **产品名 + 条款标题 + 正文**。只拿正文做向量会丢两个强信号：用户问题里常带产品名和标题词，而正文里往往不出现这些词（多个产品的「责任免除」正文高度相似，单看正文很难区分）。`content` 只保留正文，用于展示和引用。

> **注意**：`search_text` 字段必须配 `analyzer_params={"tokenizer": "jieba"}`。
> Milvus 默认分词器会把一整串中文当成一个 token，导致查"等待期"命中 0 条，
> 而查"自本合同生效之日起"这种完整长句才有结果 —— 中文 BM25 会直接废掉。

## 检索链路的三个参数

调优检索主要动这三个环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ENABLE_RERANK` | true | 关掉后耗时从约 13 秒降到约 4 秒，代价是 Top-1 命中率下降 |
| `RETRIEVAL_RERANK_POOL` | 6 | 送进重排模型的候选数。CPU 上约 2 秒/条，这是主要性能瓶颈 |
| `EMBEDDING_DEVICE` | cpu | 有 NVIDIA 显卡就改成 `cuda`，整条链路会快一个数量级 |

`POST /api/v1/agent/clause-search` 的 `mode` 参数可以切换 `hybrid` / `dense` / `sparse`，
用来对比不同检索策略的效果。

## AI 顾问（Agent）

`POST /api/v1/agent/chat` 会走一遍完整的 Agent 状态图：

```
START → 意图路由 ─┬─ clause_qa    → 条款检索 → 生成回答 → END
                  ├─ chitchat     → 生成回答 → END
                  ├─ recommend    ┐
                  ├─ premium_calc ├→ 工具调用循环 ⇄ → END
                  └─ claim_help   ┘
```

Agent 可调用的工具（定义在 `app/modules/agent/tools.py`）：

| 工具 | 作用 |
| --- | --- |
| `list_insurance_products` | 查询在售产品，支持险种和关键词筛选 |
| `calculate_premium` | 保费试算，返回完整计算过程 |
| `get_my_policies` | 查询当前用户的保单 |
| `get_my_claims` | 查询当前用户的理赔记录 |
| `get_my_orders` | 查询当前用户的订单 |
| `search_insurance_clauses` | 条款混合检索 |

每轮对话会写四张表：`conversations` / `messages` / `agent_runs` / `tool_calls`。
其中 `agent_runs` 和 `tool_calls` 是排查线上问题的依据 —— 用户投诉回答错误时，
可以还原出当时走了哪条分支、检索到哪些条款、调用了什么工具。

**耗时提示**：纯 CPU 环境下，条款问答分支约 30 秒（检索占大头），
工具调用分支约 2~3 秒。有显卡时把 `EMBEDDING_DEVICE` 改成 `cuda` 会大幅改善。

## 流式对话（SSE）

`POST /api/v1/agent/chat/stream` 采用 Server-Sent Events，事件类型如下：

| event | data | 说明 |
| --- | --- | --- |
| `meta` | `{"conversation_id": 3}` | 第一个事件，客户端要记住会话 id 才能多轮对话 |
| `status` | `{"stage": "routing", "text": "正在检索相关保险条款…"}` | 阶段提示，stage 取值 routing / tool / generating |
| `delta` | `{"text": "重疾险的"}` | 回答片段，前端追加到气泡里 |
| `done` | 含 citations / tool_calls / latency_ms / total_tokens | 结束事件 |
| `error` | `{"code": "...", "message": "..."}` | 出错 |

前端消费示例：

```js
const res = await fetch("/api/v1/agent/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
  body: JSON.stringify({ question }),
});
const reader = res.body.getReader();
const decoder = new TextDecoder();
let event = null;
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) handle(event, JSON.parse(line.slice(5)));
  }
}
```

**部署注意**：如果用 nginx 反代，必须关掉响应缓冲，否则流式会被攒成一次性返回：

```nginx
location /api/ {
    proxy_pass http://agent-service:8000;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
}
```

## 前端

React 19 + Ant Design 6 + Vite，TypeScript 严格模式。

```bash
cd frontend
npm install
npm run dev          # 开发，访问 http://localhost:5173
npm run build        # 生产构建，产物在 dist/
```

开发时 Vite 会把 `/api` 代理到 `http://127.0.0.1:8000`，所以**不需要配跨域**，
前端代码里全部用相对路径 `/api/v1/...`。

> 注意：Vite 默认只监听 IPv6 的 `localhost`。如果 `http://127.0.0.1:5173` 打不开，
> 改用 `http://localhost:5173`。

页面结构：

| 路由 | 页面 |
| --- | --- |
| `/products` | 保险商城（筛选、搜索、分页） |
| `/products/:id` | 产品详情、条款折叠、保费试算、投保下单 |
| `/orders` | 我的订单（支付 / 取消） |
| `/policies` | 我的保单 |
| `/claims` | 理赔服务（提交申请、查看进度） |
| `/login` `/register` | 登录注册 |

右下角有一个悬浮的 AI 顾问入口，**任何页面都能唤起**，它消费的是
`POST /api/v1/agent/chat/stream` 的 SSE 流。

## 一键部署

```bash
cd deploy
docker compose up -d --build
```

起来之后：

| 地址 | 说明 |
| --- | --- |
| http://localhost:8080 | 前端商城（nginx 托管 + 反向代理后端） |
| http://localhost:8000/docs | 后端接口文档 |
| http://localhost:9001 | MinIO 控制台（minioadmin / minioadmin） |

**两个前置条件**：

1. 容器里访问数据库用的是服务名（`postgres` / `redis` / `milvus`），
   这些在 `docker-compose.yml` 的 `environment` 里已经覆盖好了，
   不用改 `agent-service/.env`。
2. 后端容器需要读宿主机的模型文件，编排里挂载的是 `E:/models`。
   **如果你的模型不在 E 盘，改这一行**：

   ```yaml
   volumes:
     - E:/models:/models:ro
   ```

   同时确认 Docker Desktop 已允许访问该盘符（Settings → Resources → File sharing）。

## 数据库迁移

项目有两个数据库，用两个配置文件分别管理（共用 `alembic/env.py`）：

```bash
# 业务库 insurance_biz
uv run alembic -c alembic.ini upgrade head
uv run alembic -c alembic.ini revision --autogenerate -m "说明"

# 智能库 insurance_agent
uv run alembic -c alembic.agent.ini upgrade head
uv run alembic -c alembic.agent.ini revision --autogenerate -m "说明"
```

新增模型文件后，记得在 `app/registry.py` 里补一行导入，否则迁移会漏表。

## 启动中间件（PostgreSQL / Redis / Milvus）

```bash
cd deploy
docker compose up -d
```

首次启动 Milvus 需要 1~2 分钟初始化，稍等再看状态。
