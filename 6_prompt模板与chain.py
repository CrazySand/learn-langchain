import os

from tabulate import tabulate

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MAX_ROUNDS = 3

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ====================================================
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是{role}，我的好朋友。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)

chain = prompt | llm

history = []
# ====================================================

while True:
    user_input = input("> ").strip()
    if not user_input:
        continue
    if user_input == "exit":
        break

    # ====================================================
    payload = {
        "role": "小云",
        "history": history,
        "user_input": user_input,
    }
    system_msg_content = prompt.invoke(payload).to_messages()[0].content
    # 等价于：
    # for chunk in llm.stream(prompt.invoke(history)):
    ai_msg: AIMessageChunk | None = None
    for chunk in chain.stream(payload):
        ai_msg = chunk if ai_msg is None else ai_msg + chunk
        print(chunk.content, end="", flush=True)
    print()

    history.append(HumanMessage(content=user_input))
    history.append(ai_msg)
    history = history[-(MAX_ROUNDS * 2):]
    # ====================================================

    # ====================================================
    table: list[list] = [
        ["system", system_msg_content]
    ]
    for msg in history:
        table.append([msg.type, msg.content])
    print(tabulate(table, headers=["角色", "内容"], tablefmt="grid"))
    # ====================================================
