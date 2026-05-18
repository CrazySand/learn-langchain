import json
import os
import time

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

_HTTP_TIMEOUT = 30.0


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


llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是{role}，我的好朋友。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

llm_with_tools = llm.bind_tools(tools)

history: list = []

while True:
    user_input = input(">>>").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit"]:
        break

    payload = {
        "role": "小云",
        "history": history,
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

    history.append(HumanMessage(content=user_input))
    history.extend(generated_msgs)

    print("=" * 50)
    for msg in history:
        print(f"{msg.type}: {msg.content}")
    print("=" * 50)