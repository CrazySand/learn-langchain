import os

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    ai_msg = llm.invoke(history)
    # 等同于：history.append(AIMessage(content=ai_msg.content))
    history.append(ai_msg)
    print(ai_msg.content)
