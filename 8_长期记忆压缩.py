import os
import json
import time

from tabulate import tabulate
import httpx

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

KEEP_RAW_ROUNDS = 3
BUFFER_ROUNDS = 2

# ====================================================

_TOOL_TRANSCRIPT_MAX_LEN = 400


def _messages_to_transcript(messages: list[BaseMessage]) -> str:
    def truncate(text: str) -> str:
        if len(text) <= _TOOL_TRANSCRIPT_MAX_LEN:
            return text
        return text[: _TOOL_TRANSCRIPT_MAX_LEN] + "…(已截断)"

    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                lines.append(f"助手调用: {tc['name']}({tc['args']})")
            if m.content:
                lines.append(f"助手: {m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(f"工具结果: {truncate(m.content)}")
    return "\n".join(lines)


def flatten_rounds(rounds: list[list[BaseMessage]]) -> list[BaseMessage]:
    """将按轮存储的消息列表压平为单条消息序列。

    Args:
        rounds: 每轮一条消息列表，如 ``[Human, AI, Tool, ...]``。

    Returns:
        按轮顺序拼接后的扁平消息列表。
    """
    return [msg for r in rounds for msg in r]


def merge_into_summary(
    llm: ChatDeepSeek,
    prev_summary: str,
    fold_messages: list[BaseMessage],
) -> str:
    """把将被裁掉的对话片段合并进滚动摘要（额外一次模型调用）。

    应传入未绑定工具的摘要专用 ``llm``；内置 transcript 支持
    Human / AI（含 tool_calls）/ Tool，工具返回过长时截断。

    Args:
        prev_summary: 当前的滚动摘要文本；若尚无则为空字符串。
        fold_messages: 待折叠并入摘要的消息列表（建议为整轮 Human + generated_msgs）。

    Returns:
        合并后的新摘要字符串。模型调用失败时返回原 ``prev_summary``
        （不向外抛出异常）。
    """
    fragment = _messages_to_transcript(fold_messages)
    prompt_text = (
        "你是一个对话压缩助手。请将「已有摘要」与「新片段」合并成一份更短的中文要点列表"
        "（可编号），严格基于原文，不要编造未出现的信息。\n"
        "若片段含工具调用，须写清工具名、关键参数与结论；不要粘贴完整 JSON 或冗长原始数据。\n"
        "务必保留：数字、专名、用户明确约束与待办事项。\n\n"
        f"【已有摘要】\n{prev_summary.strip() or '（尚无）'}\n\n"
        f"【新片段】\n{fragment or '（空）'}\n\n"
        "【合并后的摘要】"
    )
    return llm.invoke(prompt_text).content


# ====================================================


@tool
def get_weather(city_name: str) -> str:
    """获取指定城市当前天气（中国气象局接口）。

    Args:
        city_name: 城市名称，中文。

    Returns:
        当前天气 JSON 字符串。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
        )
    }
    with httpx.Client(timeout=10, headers=headers, verify=False) as client:
        resp = client.get(
            "https://weather.cma.cn/api/autocomplete",
            params={
                "q": city_name,
                "limit": 1,
                "timestamp": int(time.time() * 1000),
            },
        )
        data = resp.json()
        if not data["data"]:
            raise ValueError("city not found")
        city_code = data["data"][0].split("|")[0]
        resp = client.get(f"https://weather.cma.cn/api/now/{city_code}")
        payload = resp.json()["data"]
    return json.dumps(payload, ensure_ascii=False)


@tool
def get_user_city() -> str:
    """返回演示用的用户所在城市。

    Returns:
        城市名称字符串。
    """
    return "广州"


tools = [get_weather, get_user_city]
tool_map = {tool.name: tool for tool in tools}

# ====================================================
llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    extra_body={"thinking": {"type": "disabled"}},
)

llm_with_tools = llm.bind_tools(tools)
# ====================================================


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是{role}，我的好朋友。\n\n"
            "【较早对话摘要】（仅作背景；若写「尚无」表示暂无）\n"
            "{memory}\n\n"
            "下列 history 为人类与助手最近几轮原文，请结合摘要与原文回答；不要编造摘要与原文中均未出现的事实。",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)

rounds: list[list[BaseMessage]] = []

# =========
memory = ""
# =========

while True:
    user_input = input("> ").strip()
    if not user_input:
        continue
    if user_input == "exit":
        break

    payload = {
        "role": "小云",
        # =============================
        "memory": memory or "（尚无）",
        # =============================
        "history": [msg for r in rounds for msg in r],
        "user_input": user_input,
    }
    request_msgs = prompt.invoke(payload).to_messages()
    system_msg_content = request_msgs[0].content
    generated_msgs = []

    ai_msg: AIMessageChunk | None = None
    for chunk in llm_with_tools.stream(request_msgs):
        ai_msg = chunk if ai_msg is None else ai_msg + chunk
        print(chunk.content, end="", flush=True)
    print()

    request_msgs.append(ai_msg)
    generated_msgs.append(ai_msg)

    while ai_msg.tool_calls:
        for call in ai_msg.tool_calls:
            tool_func = tool_map[call["name"]]
            tool_result = tool_func.invoke(call["args"])
            tool_msg = ToolMessage(content=tool_result,
                                   tool_call_id=call["id"])

            request_msgs.append(tool_msg)
            generated_msgs.append(tool_msg)

        ai_msg: AIMessageChunk | None = None
        for chunk in llm_with_tools.stream(request_msgs):
            ai_msg = chunk if ai_msg is None else ai_msg + chunk
            print(chunk.content, end="", flush=True)
        print()

        request_msgs.append(ai_msg)
        generated_msgs.append(ai_msg)

    rounds.append([HumanMessage(content=user_input), *generated_msgs])

    # ====================================================
    if len(rounds) > KEEP_RAW_ROUNDS + BUFFER_ROUNDS:
        fold_rounds = rounds[:-KEEP_RAW_ROUNDS]
        rounds = rounds[-KEEP_RAW_ROUNDS:]
        memory = merge_into_summary(
            llm, memory, flatten_rounds(fold_rounds)
        )
    # ====================================================

    table: list[list] = [
        ["system", system_msg_content]
    ]
    for r in rounds:
        for msg in r:
            table.append([msg.type, msg.content])
    print(tabulate(table, headers=["角色", "内容"], tablefmt="grid"))
