"""
第 1 轮结束  rounds=[1]           len=1  不压缩
第 2 轮结束  rounds=[1,2]         len=2  不压缩
第 3 轮结束  rounds=[1,2,3]       len=3  不压缩
第 4 轮结束  rounds=[1,2,3,4]     len=4  不压缩  ← BUFFER 在起作用
第 5 轮结束  rounds=[1..5]        len=5  仍不压缩（5 > 5 为假）
第 6 轮结束  len=6 > 5  → 压缩！
              fold：第 1、2、3 轮 → merge_into_summary → 写入 memory
              rounds 只剩 [4,5,6]（最近 KEEP=3 轮）

第 7、8 轮     rounds 变 4、5 轮   不压缩
第 9 轮结束  len=6  again → 再 fold 最老 3 轮进 memory，保留最近 3 轮
"""


import json
import os
import time

import httpx
from tabulate import tabulate

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

_HTTP_TIMEOUT = 30.0

_TOOL_TRANSCRIPT_MAX_LEN = 400

KEEP_RAW_ROUNDS = 3   # 折叠后仍保留原文的最近轮数
BUFFER_ROUNDS = 2     # 多攒几轮再触发摘要；超过 KEEP_RAW_ROUNDS + BUFFER_ROUNDS 时 fold


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
    with httpx.Client(timeout=_HTTP_TIMEOUT, headers=headers) as client:
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

tool_map = {
    tool.name: tool
    for tool in tools
}

# ====================================================


def merge_into_summary(
    llm: ChatDeepSeek,
    prev_summary: str,
    fold_messages: list[BaseMessage],
    *,
    tool_content_max_len: int = _TOOL_TRANSCRIPT_MAX_LEN,
) -> str:
    """把将被裁掉的对话片段合并进滚动摘要（额外一次模型调用）。

    内置 transcript：支持 Human / AI（含 tool_calls）/ Tool；工具返回过长时截断。

    Args:
        llm: 用于生成摘要的聊天模型实例。
        prev_summary: 当前的滚动摘要文本；若尚无则为空字符串。
        fold_messages: 待折叠并入摘要的消息列表（建议为整轮 Human + generated_msgs）。
        tool_content_max_len: 写入 transcript 的单条 Tool 结果最大字符数。

    Returns:
        合并后的新摘要字符串；模型调用失败时返回原 ``prev_summary``。
    """

    def _content_to_str(content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "…(已截断)"

    def _transcript(messages: list[BaseMessage]) -> str:
        lines: list[str] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                lines.append(f"用户: {_content_to_str(m.content)}")
            elif isinstance(m, AIMessage):
                for tc in m.tool_calls or []:
                    name = tc.get("name", "")
                    args = tc.get("args", {})
                    lines.append(f"助手调用: {name}({args})")
                text = _content_to_str(m.content).strip()
                if text:
                    lines.append(f"助手: {text}")
            elif isinstance(m, ToolMessage):
                body = _truncate(
                    _content_to_str(m.content), tool_content_max_len)
                lines.append(f"工具结果: {body}")
            else:
                lines.append(
                    f"{m.type}: {_truncate(_content_to_str(m.content), tool_content_max_len)}"
                )
        return "\n".join(lines)

    fragment = _transcript(fold_messages)
    prompt_text = (
        "你是一个对话压缩助手。请将「已有摘要」与「新片段」合并成一份更短的中文要点列表"
        "（可编号），严格基于原文，不要编造未出现的信息。\n"
        "若片段含工具调用，须写清工具名、关键参数与结论；不要粘贴完整 JSON 或冗长原始数据。\n"
        "务必保留：数字、专名、用户明确约束与待办事项。\n\n"
        f"【已有摘要】\n{prev_summary.strip() or '（尚无）'}\n\n"
        f"【新片段】\n{fragment or '（空）'}\n\n"
        "【合并后的摘要】"
    )
    try:
        out = llm.invoke(prompt_text)
        content = out.content
        if not content or not str(content).strip():
            return prev_summary
        return str(content).strip()
    except Exception:
        return prev_summary


def flatten_rounds(rounds: list[list[BaseMessage]]) -> list[BaseMessage]:
    return [m for turn in rounds for m in turn]

# ====================================================


llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

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
        ("human", "{question}"),
    ]
)

llm_with_tools = llm.bind_tools(tools)

memory = ""

rounds: list[list[BaseMessage]] = []

# ====================================================

while True:
    user_input = input(">>>").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit"]:
        break

    payload = {
        "role": "小云",
        "memory": memory.strip() or "（尚无）",
        "history": flatten_rounds(rounds),
        "question": user_input,
    }

    # request_msgs：每次 stream 发给模型的完整入参（system + 既往 history + 当前 question）；
    # 工具多跳时同步追加本轮 AIMessage / ToolMessage，仅供当轮请求，不直接写入 history。
    request_msgs = prompt.invoke(payload).to_messages()

    # generated_msgs：本轮用户问句之后新生成的 AIMessage / ToolMessage（可多条）；
    # 轮末与 HumanMessage 一并 append 进 history，供下一轮 prompt 的 MessagesPlaceholder 使用。
    generated_msgs = []

    ai_msg = None
    for chunk in llm_with_tools.stream(request_msgs):
        ai_msg = chunk if ai_msg is None else ai_msg + chunk
        print(chunk.content, end="", flush=True)
    print()

    request_msgs.append(ai_msg)
    generated_msgs.append(ai_msg)

    while ai_msg.tool_calls:
        for call in ai_msg.tool_calls:
            tool_fn = tool_map[call["name"]]
            tool_result = tool_fn.invoke(call["args"])
            tool_msg = ToolMessage(content=tool_result,
                                   tool_call_id=call["id"])

            request_msgs.append(tool_msg)
            generated_msgs.append(tool_msg)

        ai_msg = None
        for chunk in llm_with_tools.stream(request_msgs):
            ai_msg = chunk if ai_msg is None else ai_msg + chunk
            print(chunk.content, end="", flush=True)
        print()

        request_msgs.append(ai_msg)
        generated_msgs.append(ai_msg)

    rounds.append([HumanMessage(user_input), *generated_msgs])

    if len(rounds) > KEEP_RAW_ROUNDS + BUFFER_ROUNDS:
        fold_rounds = rounds[:-KEEP_RAW_ROUNDS]
        rounds = rounds[-KEEP_RAW_ROUNDS:]
        memory = merge_into_summary(llm, memory, flatten_rounds(fold_rounds))

    table = []
    for msg in prompt.invoke({
        "role": "小云",
        "memory": memory.strip() or "（尚无）",
        "history": flatten_rounds(rounds),
        "question": "",
    }).to_messages():
        table.append([msg.type, msg.content])
    print(tabulate(table, headers=["角色", "内容"]))
    print("=" * 50)
