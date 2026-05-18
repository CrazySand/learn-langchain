"""
带工具时，不遵守「一条 Human 必须只对应一条 AI」这种简化说法；
Human → AI（带 tool_calls）→ Tool（一条或多条）→ AI（收尾回答）.
"""

import time
import os
import json
import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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


llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)

history = [HumanMessage(content="今天深圳天气怎么样？")]
ai_msg = llm_with_tools.invoke(history)

history.append(ai_msg)

for call in ai_msg.tool_calls:
    if call["name"] == "get_weather":
        result = get_weather.invoke(call["args"])
        history.append(ToolMessage(content=result, tool_call_id=call["id"]))

ai_msg = llm_with_tools.invoke(history)

history.append(ai_msg)

print(ai_msg.content)

debug(history)
