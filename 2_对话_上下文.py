import os
from tabulate import tabulate
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 第一条通常是系统提示词
history = [SystemMessage(content="你是小明，是我的好朋友")]

max_turns = 3  # 最大回合数


def history_as_tabulate_rows(history: list, content_max_len: int = 50) -> list[dict]:
    """转为 tabulate 可用的 dict 行（headers='keys'）：role + content 在同函数内处理。"""
    rows: list[dict] = []
    for i, msg in enumerate(history):
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "human"
        elif isinstance(msg, AIMessage):
            role = "ai"
        else:
            role = type(msg).__name__

        raw = msg.content
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "↵")
        if len(raw) > content_max_len:
            raw = raw[: content_max_len - 1] + "…"

        rows.append({"#": i, "role": role, "content": raw})
    return rows


while True:
    user_input = input("> ").strip()
    if user_input == "exit":
        break

    if not user_input:
        continue

    history.append(HumanMessage(content=user_input))

    pieces: list[str] = []
    for chunk in llm.stream(history):
        if chunk.content:
            pieces.append(chunk.content)
            print(chunk.content, end="", flush=True)
    print()

    history.append(AIMessage(content="".join(pieces)))

    # --- 上下文裁剪：history[0] 为唯一 system ---

    # 写法 A（默认）：整表尾部切片；须配合 if，避免 N ≤ 2 * max_turns 时切片含 system 与 history[0] 重复
    if len(history) > 1 + max_turns * 2:
        history = [history[0], *history[-(max_turns * 2):]]

    # 本轮结束后打印当前上下文（裁剪后）
    print(tabulate(history_as_tabulate_rows(
        history), headers="keys", tablefmt="grid"))

    # 写法 B（备选，整段注释）：只裁 history[1:]，无需 if
    # system_msg = history[0]
    # rest = history[1:][-(max_turns * 2) :]
    # history = [system_msg, *rest]
