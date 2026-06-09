# learn-langchain

用 [LangChain](https://python.langchain.com/) + [DeepSeek](https://platform.deepseek.com/) 从零搭一个可对话、可流式、可调用工具、可做长期记忆压缩、可对外提供 HTTP 接口、可做 RAG 检索增强的聊天示例。本仓库以 **12 个递进式示例** 为主，建议按编号顺序学习。

---

> ## 🤖 建议配合 AI 解读
>
> 本仓库代码递进密集，**强烈建议在 Cursor、ChatGPT 等 AI 助手辅助下学习**。
>
> 推荐用法：
>
> - 打开当前课源码，让 AI **逐段解释**数据流（如 `payload` → `prep` → `prompt` → `llm`）
> - 对照上一课，让 AI 标出 **本课新增/改动** 的代码块
> - 遇到 LCEL 管道（`|`）、`itemgetter`、`RunnableParallel` 等概念时，让 AI **画图或举小例子**
> - 第 11～12 课涉及向量库与 embedding，可让 AI 解释 **建库与查询为何必须用同一模型**
>
> 可直接向 AI 提问的示例：
>
> - 「`prep | prompt | llm` 里数据是怎么传递的？」
> - 「为什么 `history` 里不放当前 `user_input`？」
> - 「`chain.stream(payload)` 和先 `invoke` 再 `llm.stream` 有什么区别？」

---

## 如何学习

### 学习顺序

**第 1～12 课环环相扣：后一个文件始终建立在前一个文件之上，请勿跳读。**


| 编号  | 文件                    | 你将学到                                                                   |
| --- | --------------------- | ---------------------------------------------------------------------- |
| 1   | `1_基本使用.ipynb`        | 初始化 `ChatDeepSeek`；同步 / 流式 / 异步调用                                      |
| 2   | `2_交互对话.py`           | 终端多轮对话；`history` 与 `HumanMessage` / `AIMessage`                        |
| 3   | `3_上下文裁剪.py`          | 保留 system + 最近 N 轮，控制上下文长度                                             |
| 4   | `4_调试历史信息.py`         | 用 `tabulate` 打印当前上下文，便于调试                                              |
| 5   | `5_流式.py`             | 流式输出；`AIMessageChunk` 合并后写入 history                                    |
| 6   | `6_prompt模板与chain.py` | `ChatPromptTemplate`、`MessagesPlaceholder`、LCEL `chain = prompt | llm` |
| 7   | `7_工具调用.py`           | `@tool`、`bind_tools`、工具调用循环（按轮保存 `rounds`）                             |
| 8   | `8_长期记忆压缩.py`         | 滚动摘要 `memory`，折叠较早轮次（`KEEP_RAW_ROUNDS` + `BUFFER_ROUNDS`）              |
| 9   | `9_Agent封装.py`        | 将第 8 课逻辑封装为 `Agent` 类；`stream` / `invoke`；`__str__` / `__repr__` 调试    |
| 10  | `10_FastAPI异步接口封装.py` | 异步 `Agent`（`astream` / `ainvoke`）；FastAPI 暴露 `/chat` 与 `/chat/stream`  |
| 11  | `11_rag基本使用.ipynb`    | PDF 加载与分块；`HuggingFaceEmbeddings`；Chroma 建库与检索                         |
| 12  | `12_rag对话.py`         | RAG 多轮对话；`prep | prompt | llm`；`itemgetter` 与检索结果整合                    |


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

建议 **Python 3.10+**（仓库在 3.12 / Conda 环境 `crazysand` 下编写）。

### 2. 安装依赖

**一键安装（推荐）：**

```bash
pip install -r requirements.txt
```

**按阶段安装：**

```bash
# 第 1～10 课
pip install langchain-deepseek langchain-core tabulate httpx

# 第 10 课及接口测试（额外）
pip install fastapi uvicorn requests

# 第 11～12 课（RAG，额外）
pip install "langchain-huggingface[full]" langchain-chroma langchain-text-splitters chromadb pypdf "sentence-transformers>=5.2.0" ipywidgets
```

> 第 11～12 课已迁移至 LangChain 新版独立包（`langchain-huggingface`、`langchain-chroma`），不再依赖 `langchain-community`。

### 3. 配置 API Key

在 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请密钥后，写入环境变量：

**Windows（PowerShell，当前会话）：**

```powershell
$env:DEEPSEEK_API_KEY = "sk-你的密钥"
```

**macOS / Linux：**

```bash
export DEEPSEEK_API_KEY="sk-你的密钥"
```

> **注意：** 若在终端里设置了 Key，再打开 Jupyter / IDE 运行脚本，需**重启内核或 IDE**，否则可能读不到环境变量。

### 4. Hugging Face 镜像（第 11 课下载 embedding 模型）

首次下载 `BAAI/bge-small-zh-v1.5` 时，建议配置国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

在 Jupyter 中可在**第一个 cell** 写入：

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

然后 **Restart Kernel** 再运行。Notebook 内核不一定继承终端环境变量。

模型下载完成后，可在代码中启用 `model_kwargs={"local_files_only": True}` 离线加载。

---

## 如何运行

### 第 1、11 课（Notebook）

```bash
jupyter notebook 1_基本使用.ipynb
jupyter notebook 11_rag基本使用.ipynb
```

或在 VS Code / Cursor 中直接打开 `.ipynb`，从上到下依次运行单元格。

**第 11 课注意：**

- 数据源：`data/员工手册.pdf`
- `Chroma.from_documents` 对同一 `collection_name` **重复调用会追加，不会覆盖**；建库只跑一次
- 建库完成后用 `Chroma(persist_directory=..., embedding_function=...)` 加载即可
- 需重建向量库时：删除 `data/chroma_db` 后重新运行建库单元格

### 第 2～9、12 课（终端脚本）

在项目根目录执行：

```bash
python 2_交互对话.py
python 3_上下文裁剪.py
# … 依此类推
python 9_Agent封装.py
python 12_rag对话.py    # 需先完成第 11 课建库
```

运行后出现 `>` 提示符，输入内容与模型对话；输入 `**exit**` 退出。

第 9 课每轮对话结束后会打印 `str(agent)` 和 `repr(agent)`，便于观察 `rounds` 与 `memory` 变化。

第 12 课每轮会先打印 `tabulate` 表格（含 system 消息与参考资料），再流式输出回答。

### 第 10 课（FastAPI 服务）

先启动服务：

```bash
python 10_FastAPI异步接口封装.py
```

默认监听 `http://0.0.0.0:8000`。接口说明：


| 方法   | 路径             | 请求体                 | 响应                 |
| ---- | -------------- | ------------------- | ------------------ |
| POST | `/chat`        | `{"message": "你好"}` | `{"reply": "..."}` |
| POST | `/chat/stream` | `{"message": "你好"}` | `text/plain` 流式文本  |


也可用 curl 快速验证：

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"你好\"}"
```

### 第 10 课接口测试（Notebook）

服务启动后，打开 `10_接口测试.ipynb`，依次运行单元格，分别测试非流式、流式与工具调用场景。

---

## 第 11～12 课：RAG 要点

### 整体流程

```
PDF → 分块 → embedding → Chroma 向量库
                              ↓
用户提问 → retriever 检索 → context 填入 system → LLM 回答
```

### 第 12 课 LCEL 结构

```python
prep = {
    "context": itemgetter("user_input") | retriever | (lambda docs: ...),
    "history": itemgetter("history"),
    "user_input": itemgetter("user_input"),
}
chain = prep | prompt | llm
```

- `**payload**`：每轮只传 `history` + `user_input`（不含 `context`）
- `**prep**`：并行检索并拼好 `context`，补全 prompt 所需变量
- `**chain.stream(payload)**`：检索 → 填模板 → 流式生成，一条链完成

> 建议用 AI 解读：让 AI 逐步解释 `itemgetter`、`prep` 并行分支、`|` 管道传值，以及为何调试表格用 `(prep | prompt).invoke` 而生成用 `chain.stream`。

### embedding 模型必须一致

第 11 课建库与第 12 课查询均使用 `BAAI/bge-small-zh-v1.5`。建库与查询模型不一致会导致检索结果失真。

---

## 推荐学习步骤

1. **先跑通第 1 课**，确认 API Key 与网络正常。
2. **打开当前课与上一课**，对照分隔注释，看本课多了哪些行、改了哪些逻辑；**不懂就问 AI**。
3. **自己改一两处做实验**（例如改 `MAX_ROUNDS`、`K` 检索条数、改 system 人设），观察 `tabulate` 表格或终端输出的变化。
4. **第 6 课起**注意：`history` / `rounds` 里只放已完成的轮次，当前用户输入通过模板变量 `user_input` 传入，避免重复 append。
5. **第 7～10 课**会调用天气相关工具；模型使用 `deepseek-v4-flash`，并已关闭 thinking 模式，避免工具场景下的 API 兼容问题。
6. **第 9 课**把脚本式逻辑收进 `Agent` 类，对外只需 `agent.stream(...)` 或 `agent.invoke(...)`。
7. **第 10 课**在类内部用 `asyncio.Lock` 串行化对话，避免并发请求同时改写 `rounds` / `memory`。
8. **第 11 课**完成向量库建库后，再运行 **第 12 课** RAG 对话；可用「手册里什么时候发工资」等问题验证检索效果。

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
├── 11_rag基本使用.ipynb          # RAG 建库与检索
├── 12_rag对话.py                 # RAG 多轮对话（LCEL 完整链）
├── data/
│   ├── 员工手册.pdf              # RAG 数据源
│   ├── chroma_db/                # 向量库（本地生成，已 gitignore）
│   └── rag_result_demo.txt       # 检索结果示例
├── requirements.txt
└── README.md
```

---

## 常见问题


| 现象                                        | 可能原因                        | 处理                                                                  |
| ----------------------------------------- | --------------------------- | ------------------------------------------------------------------- |
| `DEEPSEEK_API_KEY must be set`            | 当前进程未读到环境变量                 | 设置 Key 后重启终端 / Jupyter 内核                                           |
| `reasoning_content must be passed back`   | thinking 模式 + 工具调用          | 第 7～10 课已禁用 thinking；勿去掉 `extra_body`                               |
| 天气工具 SSL 报错                               | 气象站证书问题                     | 示例已 `verify=False`；或改用 mock 数据                                      |
| 表格里出现 `AIMessageChunk`                    | 流式合并后未转为 `AIMessage`        | 不影响单轮工具链；若要规范可改为 `AIMessage` 再写入 history                            |
| 第 10 课 Notebook 连接失败                      | 服务未启动或端口不对                  | 先运行 `10_FastAPI异步接口封装.py`，确认 `BASE_URL` 为 `http://127.0.0.1:8000`   |
| 并发请求时对话状态错乱                               | 多请求同时修改同一 `agent`           | 第 10 课已在 `astream` 内加锁；生产环境建议按用户 / 会话隔离 Agent 实例                    |
| HuggingFace 连不上 / 仍访问 huggingface.co      | Jupyter 内核未读到 `HF_ENDPOINT` | 在 Notebook 第一个 cell 设置 `os.environ["HF_ENDPOINT"]` 并 Restart Kernel |
| RAG 检索结果离谱                                | 建库与查询 embedding 模型不一致       | 统一使用 `BAAI/bge-small-zh-v1.5`                                       |
| Chroma 检索大量重复                             | 重复执行 `from_documents`       | 建库只跑一次；重建时先删 `data/chroma_db`                                       |
| `LangChainDeprecationWarning` (community) | 旧版 import 路径                | 第 11～12 课已改用 `langchain-huggingface` / `langchain-chroma`           |


---

## 许可与声明

示例代码仅供学习；外网 API 与第三方站点可用性会变化，请以官方文档为准。