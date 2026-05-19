import os

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MAX_ROUNDS = 3

# 实例化客户端对象
client = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat"
)


# 定义模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你叫做{name}，是我的好朋友。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}")
    ]
)

# 定义历史对话
history = []

# 进入循环
while True:
    # 在终端接收用户输入，并且去掉前后的空格
    user_input = input(">>>").strip()
    # 如果用户输入了 exit/quit，则退出循环
    if user_input.lower() in ["exit", "quit"]:
        break
    # 如果用户输入了空字符串，则跳过本次循环
    if not user_input:
        continue

    # 本次请求的载荷（参数）
    payload = {
        "name": "小明",
        "history": history,
        "user_input": user_input
    }
    # 传入载荷，获取模板值对象
    obj = prompt.invoke(payload)
    # 将模板值对象转换为消息列表
    request_msgs = obj.to_messages()

    # 定义一个列表，用来存储 AI 的流式片段字符串回复
    ai_messages_pieces = []
    # 遍历流式响应，获取 AI 的流式片段字符串回复
    for chunk in client.stream(request_msgs):
        # 如果此次流式响应有内容，则将内容添加到列表中，并且打印
        if chunk.content:
            ai_messages_pieces.append(chunk.content)
            print(chunk.content, end="", flush=True)
    print()

    # 流式对话结束，将用户输入和 AI 的流式片段字符串回复添加到历史对话中
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content="".join(ai_messages_pieces)))

    # 截取历史对话，只保留最近的 MAX_ROUNDS * 2 条
    history = history[-MAX_ROUNDS * 2:]

    # 打印历史对话
    print("=" * 100)
    for msg in history:
        print(f"{msg.type}: {msg .content}")
    print("=" * 100)
