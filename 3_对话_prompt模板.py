import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek

MAX_RAW_ROUNDS = 3

llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是{role}，我的好朋友"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# LCEL：先渲染模板，再把 messages 交给同一套 llm（与非链式的 prompt.invoke + llm.stream 等价）
chain = prompt | llm

# 只存「已完成」的多轮消息（HumanMessage / AIMessage）；当前这一句用 {question}，不要提前塞进 history
history = []


def transcript(messages: list) -> str:
    lines: list[str] = []
    for m in messages:
        tag = type(m).__name__.replace("Message", "").lower()
        body = m.content
        lines.append(f"{tag}: {body}")
    return "\n".join(lines)


def debug(messages: list):
    with open("debug.txt", "w",) as f:
        for idx, msg in enumerate(messages):
            f.write(f"{idx}-{msg.type}: {msg.content}\n")
            f.write("=" * 50 + "\n")


while True:
    user_input = input("> ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    debug(history)

    payload = {
        "role": "小明",
        "history": history,
        "question": user_input,
    }

    pieces: list[str] = []
    for chunk in chain.stream(payload):
        if chunk.content:
            pieces.append(chunk.content)
            print(chunk.content, end="", flush=True)
    print()

    # 非流式等价写法：ai_msg = chain.invoke(payload)

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content="".join(pieces)))

    # 只保留最近的 n 轮对话
    history = history[-(MAX_RAW_ROUNDS * 2):]
