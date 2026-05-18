import os
import time
import json
import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek


@tool
def get_weather(city_name: str) -> str:
    """获取指定城市天气数据"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"}
    resp = httpx.get(
        url="https://weather.cma.cn/api/autocomplete",
        params={"q": city_name, "limit": 1,
                "timestamp": int(time.time() * 1000)},
        headers=headers
    )
    data = resp.json()
    if not data['data']:
        raise ValueError("city not found")
    city_code = data['data'][0].split('|')[0]
    resp = httpx.get(
        url=f"https://weather.cma.cn/api/now/{city_code}",
        headers=headers
    )
    data = resp.json()["data"]
    return json.dumps(data, ensure_ascii=False)


llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

tools = [get_weather]

tool_map = {
    "get_weather": get_weather,
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是{role}，我的好朋友"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm.bind_tools(tools)

history = []


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

    payload = {
        "role": "小明",
        "history": history,
        "question": user_input,
    }
    debug(prompt.invoke(payload).to_messages())

    ai_msg = chain.invoke(payload)

    if ai_msg.tool_calls:
        history.append(HumanMessage(content=user_input))
        history.append(ai_msg)

        for call in ai_msg.tool_calls:
            tool = tool_map[call["name"]]
            result = tool.invoke(call["args"])
            history.append(ToolMessage(
                content=result, tool_call_id=call["id"]))

        ai_msg = chain.invoke(payload)
        history.append(ai_msg)
    else:
        history.append(HumanMessage(content=user_input))
        history.append(ai_msg)

    print(ai_msg.content)
