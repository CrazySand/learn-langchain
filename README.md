# learn-langchain

用 [LangChain](https://python.langchain.com/) + [DeepSeek](https://platform.deepseek.com/) 从零搭一个可对话、可流式、可调用工具、可做长期记忆压缩、可对外提供 HTTP 接口的聊天示例。本仓库以 **10 个递进式示例** 为主，建议按编号顺序学习。

## 如何学习

### 学习顺序

**第 1～10 课环环相扣：后一个文件始终建立在前一个文件之上，请勿跳读。**

| 编号 | 文件 | 你将学到 |
|------|------|----------|
| 1 | `1_基本使用.ipynb` | 初始化 `ChatDeepSeek`；同步 / 流式 / 异步调用 |
| 2 | `2_交互对话.py` | 终端多轮对话；`history` 与 `HumanMessage` / `AIMessage` |
| 3 | `3_上下文裁剪.py` | 保留 system + 最近 N 轮，控制上下文长度 |
| 4 | `4_调试历史信息.py` | 用 `tabulate` 打印当前上下文，便于调试 |
| 5 | `5_流式.py` | 流式输出；`AIMessageChunk` 合并后写入 history |
| 6 | `6_prompt模板与chain.py` | `ChatPromptTemplate`、`MessagesPlaceholder`、LCEL `chain = prompt \| llm` |
| 7 | `7_工具调用.py` | `@tool`、`bind_tools`、工具调用循环（按轮保存 `rounds`） |
| 8 | `8_长期记忆压缩.py` | 滚动摘要 `memory`，折叠较早轮次（`KEEP_RAW_ROUNDS` + `BUFFER_ROUNDS`） |
| 9 | `9_Agent封装.py` | 将第 8 课逻辑封装为 `Agent` 类；`stream` / `invoke`；`__str__` / `__repr__` 调试 |
| 10 | `10_FastAPI异步接口封装.py` | 异步 `Agent`（`astream` / `ainvoke`）；FastAPI 暴露 `/chat` 与 `/chat/stream` |

配套：`10_接口测试.ipynb` 用 `requests` 调用第 10 课接口，可在学完第 10 课后运行。

### 相对上一课改了什么

每课相对上一课**有新增或调整**的代码块，会用分隔注释标出，便于对照：

```python
# =========================================================
```

在编辑器中搜索 `# =====` 或 `================================================` 可快速跳到本课改动处。

### 代码风格说明

示例在**变量命名、逻辑结构、写法**上保持统一、克制，方便你专注理解 LangChain，而不是纠结风格差异。同一概念在不同课中尽量沿用相同名称（如 `history`、`payload`、`rounds`、`memory`）。

---

## 环境准备

### 1. Python

建议 **Python 3.10+**（仓库在 3.12 下编写）。

### 2. 安装依赖

**第 1～9 课：**

```bash
pip install langchain-deepseek langchain-core tabulate httpx
```

**第 10 课及接口测试（额外）：**

```bash
pip install fastapi uvicorn requests
```

第 7～10 课的天气工具会请求外网；若证书报错，示例里对气象接口使用了 `verify=False`，仅用于本地演示。

### 3. 配置 API Key

在 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请密钥后，写入环境变量：

**Windows（PowerShell，当前会话）：**

```powershell
$env:DEEPSEEK_API_KEY = "sk-你的密钥"
```

**Windows（永久）：** 系统环境变量中新建 `DEEPSEEK_API_KEY`。

**macOS / Linux：**

```bash
export DEEPSEEK_API_KEY="sk-你的密钥"
```

> **注意：** 若在终端里设置了 Key，再打开 Jupyter / IDE 运行脚本，需**重启内核或 IDE**，否则可能读不到环境变量。

---

## 如何运行

### 第 1 课（Notebook）

```bash
jupyter notebook 1_基本使用.ipynb
```

或在 VS Code / Cursor 中直接打开该 `.ipynb`，从上到下依次运行单元格。

### 第 2～9 课（终端脚本）

在项目根目录执行：

```bash
python 2_交互对话.py
python 3_上下文裁剪.py
# … 依此类推
python 9_Agent封装.py
```

运行后终端出现 `>` 提示符，输入内容与模型对话；输入 **`exit`** 退出。

第 9 课每轮对话结束后会打印 `str(agent)`（当前上下文表格）和 `repr(agent)`（实例摘要），便于观察 `rounds` 与 `memory` 变化。

### 第 10 课（FastAPI 服务）

先启动服务：

```bash
python 10_FastAPI异步接口封装.py
```

默认监听 `http://0.0.0.0:8000`。接口说明：

| 方法 | 路径 | 请求体 | 响应 |
|------|------|--------|------|
| POST | `/chat` | `{"message": "你好"}` | `{"reply": "..."}` |
| POST | `/chat/stream` | `{"message": "你好"}` | `text/plain` 流式文本 |

也可用 curl 快速验证：

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"你好\"}"
```

### 第 10 课接口测试（Notebook）

服务启动后，打开 `10_接口测试.ipynb`，依次运行单元格，分别测试非流式、流式与工具调用场景。

---

## 推荐学习步骤

1. **先跑通第 1 课**，确认 API Key 与网络正常。
2. **打开当前课与上一课**，对照分隔注释，看本课多了哪些行、改了哪些逻辑。
3. **自己改一两处做实验**（例如改 `MAX_ROUNDS`、`keep_raw_rounds`、改 system 人设），观察 `tabulate` 表格或终端输出的变化。
4. **第 6 课起**注意：`history` / `rounds` 里只放已完成的轮次，当前用户输入通过模板变量 `user_input` 传入，避免重复 append。
5. **第 7～10 课**会调用天气相关工具；模型使用 `deepseek-v4-flash`，并已关闭 thinking 模式（`extra_body={"thinking": {"type": "disabled"}}`），避免工具场景下的 API 兼容问题。
6. **第 9 课**把脚本式逻辑收进 `Agent` 类，对外只需 `agent.stream(...)` 或 `agent.invoke(...)`。
7. **第 10 课**在类内部用 `asyncio.Lock` 串行化对话，避免并发请求同时改写 `rounds` / `memory`；`ainvoke` 复用 `astream` 收集完整回复。

---

## 目录说明

```
learn-langchain/
├── 1_基本使用.ipynb              # 模型调用入门
├── 2_交互对话.py                 # 多轮对话
├── 3_上下文裁剪.py               # 最近 N 轮裁剪
├── 4_调试历史信息.py             # tabulate 调试
├── 5_流式.py                     # 流式输出
├── 6_prompt模板与chain.py        # Prompt 模板 + LCEL
├── 7_工具调用.py                 # 工具调用循环
├── 8_长期记忆压缩.py             # 滚动摘要 memory
├── 9_Agent封装.py                # Agent 类（同步）
├── 10_FastAPI异步接口封装.py     # 异步 Agent + FastAPI
├── 10_接口测试.ipynb             # 接口测试（配合第 10 课）
└── README.md
```

---

## 常见问题

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `DEEPSEEK_API_KEY must be set` | 当前进程未读到环境变量 | 设置 Key 后重启终端 / Jupyter 内核 |
| `reasoning_content must be passed back` | thinking 模式 + 工具调用 | 第 7～10 课已禁用 thinking；勿去掉 `extra_body` |
| 天气工具 SSL 报错 | 气象站证书问题 | 示例已 `verify=False`；或改用 mock 数据 |
| 表格里出现 `AIMessageChunk` | 流式合并后未转为 `AIMessage` | 不影响单轮工具链；若要规范可改为 `AIMessage` 再写入 history |
| 第 10 课 Notebook 连接失败 | 服务未启动或端口不对 | 先运行 `python 10_FastAPI异步接口封装.py`，确认 `BASE_URL` 为 `http://127.0.0.1:8000` |
| 并发请求时对话状态错乱 | 多请求同时修改同一 `agent` | 第 10 课已在 `astream` 内加锁；生产环境建议按用户 / 会话隔离 Agent 实例 |

---

## 许可与声明

示例代码仅供学习；外网 API 与第三方站点可用性会变化，请以官方文档为准。
