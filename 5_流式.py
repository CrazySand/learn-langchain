import os

from tabulate import tabulate

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage

MAX_ROUNDS = 3

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

history = [SystemMessage(content="你是小云，是我的好朋友。")]

while True:
    user_input = input("> ").strip()
    if not user_input:
        continue
    if user_input == "exit":
        break

    history.append(HumanMessage(content=user_input))

    # ===========================================
    ai_msg: AIMessageChunk | None = None
    for chunk in llm.stream(history):
        ai_msg = chunk if ai_msg is None else ai_msg + chunk
        print(chunk.content, end="", flush=True)
    print()
    # ===========================================

    history.append(ai_msg)

    if len(history) > 1 + MAX_ROUNDS * 2:
        history = [history[0], *history[-(MAX_ROUNDS * 2):]]

    table: list[list] = []
    for msg in history:
        table.append([msg.type, msg.content])
    print(tabulate(table, headers=["角色", "内容"], tablefmt="grid"))
