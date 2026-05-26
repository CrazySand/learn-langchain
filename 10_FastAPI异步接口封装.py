import asyncio
import os
import json
import time
from typing import AsyncGenerator

from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse
from tabulate import tabulate
import httpx

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, Tool


class Agent:

    def __init__(
        self,
        *,
        deepseek_api_key: str,
        friend_name: str,
        tools: list[Tool],
        keep_raw_rounds: int = 3,
        buffer_rounds: int = 2,
    ) -> None:
        self.llm = ChatDeepSeek(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key=deepseek_api_key,
            extra_body={"thinking": {"type": "disabled"}},
        )
        self.llm_with_tools = self.llm.bind_tools(tools)

        self.tool_map = {tool.name: tool for tool in tools}

        self.friend_name = friend_name

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是{role}，我的好朋友。\n\n"
                    "【较早对话摘要】（仅作背景；若写「尚无」表示暂无）\n"
                    "{memory}\n\n"
                    "下列 history 为人类与助手最近几轮原文，请结合摘要与原文回答；不要编造摘要与原文中均未出现的事实。",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_input}"),
            ]
        )
        self.rounds: list[list[BaseMessage]] = []
        self.memory = ""

        self.keep_raw_rounds = keep_raw_rounds
        self.buffer_rounds = buffer_rounds

        self._tool_transcript_max_len = 400
        self._chat_lock = asyncio.Lock()

    async def astream(self, user_input: str) -> AsyncGenerator[str, None]:
        async with self._chat_lock:
            payload = {
                "role": self.friend_name,
                "memory": self.memory or "（尚无）",
                "history": [msg for r in self.rounds for msg in r],
                "user_input": user_input,
            }
            request_msgs = self.prompt.invoke(payload).to_messages()
            generated_msgs = []

            ai_msg: AIMessageChunk | None = None
            async for chunk in self.llm_with_tools.astream(request_msgs):
                ai_msg = chunk if ai_msg is None else ai_msg + chunk
                yield chunk.content

            request_msgs.append(ai_msg)
            generated_msgs.append(ai_msg)

            while ai_msg.tool_calls:
                for call in ai_msg.tool_calls:
                    tool_func = self.tool_map[call["name"]]
                    tool_result = await tool_func.ainvoke(call["args"])
                    tool_msg = ToolMessage(
                        content=tool_result,
                        tool_call_id=call["id"],
                    )

                    request_msgs.append(tool_msg)
                    generated_msgs.append(tool_msg)

                ai_msg = None
                async for chunk in self.llm_with_tools.astream(request_msgs):
                    ai_msg = chunk if ai_msg is None else ai_msg + chunk
                    yield chunk.content

                request_msgs.append(ai_msg)
                generated_msgs.append(ai_msg)

            self.rounds.append(
                [HumanMessage(content=user_input), *generated_msgs])

            if len(self.rounds) > self.keep_raw_rounds + self.buffer_rounds:
                fold_rounds = self.rounds[: -self.keep_raw_rounds]
                self.rounds = self.rounds[-self.keep_raw_rounds:]
                self.memory = await self.amerge_into_summary(
                    self.memory, self.flatten_rounds(fold_rounds)
                )
        print(f"\n{str(self)}\n{repr(self)}\n")

    async def ainvoke(self, user_input: str) -> str:
        parts: list[str] = []
        async for piece in self.astream(user_input):
            if piece:
                parts.append(piece)
        return "".join(parts)

    def _messages_to_transcript(self, messages: list[BaseMessage]) -> str:
        def truncate(text: str) -> str:
            if len(text) <= self._tool_transcript_max_len:
                return text
            return text[: self._tool_transcript_max_len] + "…(已截断)"

        lines: list[str] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                lines.append(f"用户: {m.content}")
            elif isinstance(m, AIMessage):
                for tc in m.tool_calls or []:
                    lines.append(f"助手调用: {tc['name']}({tc['args']})")
                if m.content:
                    lines.append(f"助手: {m.content}")
            elif isinstance(m, ToolMessage):
                lines.append(f"工具结果: {truncate(m.content)}")
        return "\n".join(lines)

    @staticmethod
    def flatten_rounds(rounds: list[list[BaseMessage]]) -> list[BaseMessage]:
        """将按轮存储的消息列表压平为单条消息序列。

        Args:
            rounds: 每轮一条消息列表，如 ``[Human, AI, Tool, ...]``。

        Returns:
            按轮顺序拼接后的扁平消息列表。
        """
        return [msg for r in rounds for msg in r]

    async def amerge_into_summary(
        self,
        prev_summary: str,
        fold_messages: list[BaseMessage],
    ) -> str:
        """把将被裁掉的对话片段合并进滚动摘要（额外一次模型调用）。

        应传入未绑定工具的摘要专用 ``llm``；内置 transcript 支持
        Human / AI（含 tool_calls）/ Tool，工具返回过长时截断。

        Args:
            prev_summary: 当前的滚动摘要文本；若尚无则为空字符串。
            fold_messages: 待折叠并入摘要的消息列表（建议为整轮 Human + generated_msgs）。

        Returns:
            合并后的新摘要字符串。模型调用失败时返回原 ``prev_summary``
            （不向外抛出异常）。
        """
        fragment = self._messages_to_transcript(fold_messages)
        prompt_text = (
            "你是一个对话压缩助手。请将「已有摘要」与「新片段」合并成一份更短的中文要点列表"
            "（可编号），严格基于原文，不要编造未出现的信息。\n"
            "若片段含工具调用，须写清工具名、关键参数与结论；不要粘贴完整 JSON 或冗长原始数据。\n"
            "务必保留：数字、专名、用户明确约束与待办事项。\n\n"
            f"【已有摘要】\n{prev_summary.strip() or '（尚无）'}\n\n"
            f"【新片段】\n{fragment or '（空）'}\n\n"
            "【合并后的摘要】"
        )
        return (await self.llm.ainvoke(prompt_text)).content

    def __str__(self) -> str:
        msgs = self.prompt.invoke({
            "role": self.friend_name,
            "memory": self.memory or "（尚无）",
            "history": [msg for r in self.rounds for msg in r],
            "user_input": "",
        }).to_messages()
        table: list[list[str, str]] = []
        for msg in msgs:
            table.append([msg.type, msg.content])
        return tabulate(table, headers=["类型", "内容"], tablefmt="grid")

    def __repr__(self) -> str:
        return (
            f"Agent("
            f"tools_count={len(self.tool_map)}, "
            f"friend_name={self.friend_name!r}, "
            f"rounds_count={len(self.rounds)}, "
            f"memory_len={len(self.memory)}, "
            f"keep_raw_rounds={self.keep_raw_rounds}, "
            f"buffer_rounds={self.buffer_rounds}"
            f")"
        )

# ==================================================


@tool
async def get_weather(city_name: str) -> str:
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
    async with httpx.AsyncClient(timeout=10, headers=headers, verify=False) as client:
        resp = await client.get(
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
        resp = await client.get(f"https://weather.cma.cn/api/now/{city_code}")
        payload = resp.json()["data"]
    return json.dumps(payload, ensure_ascii=False)


@tool
async def get_user_city() -> str:
    """返回演示用的用户所在城市。

    Returns:
        城市名称字符串。
    """
    return "广州"


# =========================================================


agent = Agent(
    deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    friend_name="小云",
    tools=[get_weather, get_user_city],
    keep_raw_rounds=3,
    buffer_rounds=2,
)

app = FastAPI()


@app.post("/chat/stream")
async def chat_stream(message: str = Body(..., embed=True)) -> StreamingResponse:
    """流式聊天：封装 ``Agent.astream``，按纯文本分块返回。"""

    async def text_stream() -> AsyncGenerator[str, None]:
        async for piece in agent.astream(message):
            if piece:
                yield piece

    return StreamingResponse(text_stream(), media_type="text/plain; charset=utf-8")


@app.post("/chat")
async def chat(message: str = Body(..., embed=True)) -> dict[str, str]:
    """非流式聊天：封装 ``Agent.ainvoke``，一次返回完整回复。"""
    reply = await agent.ainvoke(message)
    return {"reply": reply}


# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
