# learn-langchain

用 [LangChain](https://python.langchain.com/) + [DeepSeek](https://platform.deepseek.com/) 从零搭一个可对话、可流式、可调用工具、可做长期记忆压缩的聊天示例。本仓库以 **9 个递进式示例** 为主，建议按编号顺序学习。

## 如何学习

### 学习顺序

**第 1～9 课环环相扣：后一个文件始终建立在前一个文件之上，请勿跳读。**

| 编号 | 文件 | 你将学到 |
|------|------|----------|
| 1 | `1_基本使用.ipynb` | 初始化 `ChatDeepSeek`；同步 / 流式 / 异步调用 |
| 2 | `2_交互对话.py` | 终端多轮对话；`history` 与 `HumanMessage` / `AIMessage` |
| 3 | `3_上下文裁剪.py` | 保留 system + 最近 N 轮，控制上下文长度 |
| 4 | `4_调试历史信息.py` | 用 `tabulate` 打印当前上下文，便于调试 |
| 5 | `5_流式.py` | 流式输出；边收 token 边打印 |
| 6 | `6_prompt模板与chain.py` | `ChatPromptTemplate`、`MessagesPlaceholder`、LCEL `chain = prompt \| llm` |
| 7 | `7_工具调用.py` | `@tool`、`bind_tools`、工具调用循环（按轮保存 history） |
| 8 | `8_长期记忆压缩.py` | 在工具对话基础上增加滚动摘要 `memory`，折叠较早轮次 |
| 9 | `9_封装.py` | 将前述能力封装为类（编写中） |

### 相对上一课改了什么

每课相对上一课**有新增或调整**的代码块，会用分隔注释标出，便于对照：

```python
# =========================================================
```

在编辑器中搜索 `# =====` 或 `================================================` 可快速跳到本课改动处。

### 代码风格说明

示例在**变量命名、逻辑结构、写法**上保持统一、克制，方便你专注理解 LangChain，而不是纠结风格差异。同一概念在不同课中尽量沿用相同名称（如 `history`、`payload`、`rounds`）。

---

## 环境准备

### 1. Python

建议 **Python 3.10+**（仓库在 3.12 下编写）。

### 2. 安装依赖

```bash
pip install langchain-deepseek langchain-core tabulate httpx
```

第 7、8 课的天气工具会请求外网；若证书报错，示例里对气象接口使用了 `verify=False`，仅用于本地演示。

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

### 第 2～8 课（脚本）

在项目根目录执行：

```bash
python 2_交互对话.py
python 3_上下文裁剪.py
# … 依此类推
python 8_长期记忆压缩.py
```

运行后终端出现 `>` 提示符，输入内容与模型对话；输入 **`exit`** 退出。

### 第 9 课

`9_封装.py` 尚未完成，学完第 8 课后可先阅读 `新建文件夹/` 中的参考实现（见下文）。

---

## 推荐学习步骤

1. **先跑通第 1 课**，确认 API Key 与网络正常。
2. **打开当前课与上一课**，对照分隔注释，看本课多了哪些行、改了哪些逻辑。
3. **自己改一两处做实验**（例如改 `MAX_ROUNDS`、改 system 人设），观察 `tabulate` 表格或终端输出的变化。
4. **第 6 课起**注意：`history` 里只放已完成的轮次，当前用户输入通过模板变量 `user_input` 传入，避免重复 append。
5. **第 7、8 课**会调用天气相关工具；模型使用 `deepseek-v4-flash`，并已关闭 thinking 模式（`extra_body={"thinking": {"type": "disabled"}}`），避免工具场景下的 API 兼容问题。

---

## 目录说明

```
learn-langchain/
├── 1_基本使用.ipynb      # 模型调用入门
├── 2_交互对话.py         # 多轮对话
├── …
├── 8_长期记忆压缩.py     # 摘要 + 工具 + 流式
├── 9_封装.py             # 封装（待完成）
├── 新建文件夹/           # 早期版本与更完整参考实现，非主学习路径
└── README.md
```

`新建文件夹/` 内另有更完整的封装、异步接口等示例，可在掌握 1～8 后作扩展阅读，**不必按该目录文件名顺序学习**。

---

## 常见问题

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `DEEPSEEK_API_KEY must be set` | 当前进程未读到环境变量 | 设置 Key 后重启终端 / Jupyter 内核 |
| `reasoning_content must be passed back` | thinking 模式 + 工具调用 | 第 7、8 课已禁用 thinking；勿去掉 `extra_body` |
| 天气工具 SSL 报错 | 气象站证书问题 | 示例已 `verify=False`；或改用 mock 数据 |
| 表格里出现 `AIMessageChunk` | 流式合并后未转为 `AIMessage` | 不影响单轮工具链；若要规范可改为 `AIMessage` 再写入 history |

---

## 许可与声明

示例代码仅供学习；外网 API 与第三方站点可用性会变化，请以官方文档为准。
