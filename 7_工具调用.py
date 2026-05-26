import os
import json
import time

from tabulate import tabulate
import httpx

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

MAX_ROUNDS = 3


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


llm_with_tools = ChatDeepSeek(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    extra_body={"thinking": {"type": "disabled"}},
).bind_tools(tools)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是{role}，我的好朋友。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)

rounds: list[list[BaseMessage]] = []

while True:
    user_input = input("> ").strip()
    if not user_input:
        continue
    if user_input == "exit":
        break

    payload = {
        "role": "小云",
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

    rounds = rounds[-(MAX_ROUNDS):]

    table: list[list] = [
        ["system", system_msg_content]
    ]
    for r in rounds:
        for msg in r:
            table.append([msg.type, msg.content])
    print(tabulate(table, headers=["角色", "内容"], tablefmt="grid"))
