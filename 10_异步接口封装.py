import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse
from tabulate import tabulate

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, tool
from langchain_deepseek import ChatDeepSeek


class MyAgent:

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://api.deepseek.com",
        friend_name: str,
        tools: list[Tool],
        keep_raw_rounds: int = 5,
        buffer_rounds: int = 3,
    ):
        base_llm = ChatDeepSeek(
            model=model,
            base_url=base_url,
            api_key=api_key,
            extra_body={"thinking": {"type": "disabled"}},
        )
        self.llm = base_llm.bind_tools(tools)
        self._summary_llm = base_llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你的名字是{role}，是用户的好朋友。\n\n"
                    "【较早对话摘要】（仅作背景；若写「尚无」表示暂无）\n"
                    "{memory}\n\n"
                    "下列 history 为人类与助手最近几轮原文，请结合摘要与原文回答；"
                    "不要编造摘要与原文中均未出现的事实。",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        self.keep_raw_rounds = keep_raw_rounds
        self.buffer_rounds = buffer_rounds

        self.friend_name = friend_name
        self.tool_map = {
            tool.name: tool
            for tool in tools
        }

        self.memory = ""
        self.rounds: list[list[BaseMessage]] = []

        self._tool_content_max_len = 400

    async def _finalize_turn(
        self, question: str, generated_msgs: list[BaseMessage]
    ) -> None:
        """写入本轮消息并在超出阈值时折叠进 memory。"""
        self.rounds.append([HumanMessage(content=question), *generated_msgs])
        if len(self.rounds) > self.keep_raw_rounds + self.buffer_rounds:
            fold_rounds = self.rounds[:-self.keep_raw_rounds]
            self.rounds = self.rounds[-self.keep_raw_rounds:]
            self.memory = await self.amerge_into_summary(
                self.memory, self.flatten_rounds(fold_rounds)
            )

    async def astream(self, question: str) -> AsyncIterator[str]:
        """流式执行一轮对话，逐块 yield 助手文本片段。

        Args:
            question: 用户输入。

        Yields:
            模型输出的文本片段。
        """
        request_msgs = self.prompt.invoke(
            self._chat_payload(question)
        ).to_messages()
        generated_msgs: list[BaseMessage] = []

        ai_msg = None
        async for chunk in self.llm.astream(request_msgs):
            ai_msg = chunk if ai_msg is None else ai_msg + chunk
            if chunk.content:
                yield chunk.content

        request_msgs.append(ai_msg)
        generated_msgs.append(ai_msg)

        while ai_msg and ai_msg.tool_calls:
            for call in ai_msg.tool_calls:
                tool_fn = self.tool_map[call["name"]]
                tool_result = await tool_fn.ainvoke(call["args"])
                tool_msg = ToolMessage(
                    content=tool_result, tool_call_id=call["id"]
                )
                request_msgs.append(tool_msg)
                generated_msgs.append(tool_msg)

            ai_msg = None
            async for chunk in self.llm.astream(request_msgs):
                ai_msg = chunk if ai_msg is None else ai_msg + chunk
                if chunk.content:
                    yield chunk.content

            request_msgs.append(ai_msg)
            generated_msgs.append(ai_msg)

        await self._finalize_turn(question, generated_msgs)

    async def ainvoke(self, question: str) -> str:
        """非流式执行一轮对话，返回本轮助手可见文本。

        Args:
            question: 用户输入。

        Returns:
            本轮各段 AIMessage 的 content 拼接结果；无文本时为空字符串。
        """
        request_msgs = self.prompt.invoke(
            self._chat_payload(question)
        ).to_messages()
        generated_msgs: list[BaseMessage] = []
        pieces: list[str] = []

        ai_msg = await self.llm.ainvoke(request_msgs)
        if ai_msg.content:
            pieces.append(ai_msg.content)
        request_msgs.append(ai_msg)
        generated_msgs.append(ai_msg)

        while ai_msg.tool_calls:
            for call in ai_msg.tool_calls:
                tool_fn = self.tool_map[call["name"]]
                tool_result = await tool_fn.ainvoke(call["args"])
                tool_msg = ToolMessage(
                    content=tool_result, tool_call_id=call["id"]
                )
                request_msgs.append(tool_msg)
                generated_msgs.append(tool_msg)

            ai_msg = await self.llm.ainvoke(request_msgs)
            if ai_msg.content:
                pieces.append(ai_msg.content)
            request_msgs.append(ai_msg)
            generated_msgs.append(ai_msg)

        await self._finalize_turn(question, generated_msgs)
        return "".join(pieces)

    async def amerge_into_summary(
        self,
        prev_summary: str,
        fold_messages: list[BaseMessage],
    ) -> str:
        """把将被裁掉的对话片段合并进滚动摘要（额外一次模型调用）。

        使用未绑定工具的 ``self._summary_llm`` 生成摘要；内置 transcript 支持
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
        try:
            out = await self._summary_llm.ainvoke(
                [HumanMessage(content=prompt_text)]
            )
            return out.content.strip()
        except Exception:
            logging.exception("amerge_into_summary failed")
            return prev_summary

    def _messages_to_transcript(self, messages: list[BaseMessage]) -> str:
        def truncate(text: str) -> str:
            if len(text) <= self._tool_content_max_len:
                return text
            return text[: self._tool_content_max_len] + "…(已截断)"

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

    def _chat_payload(self, question: str) -> dict:
        return {
            "role": self.friend_name,
            "memory": self.memory.strip() or "（尚无）",
            "history": self.flatten_rounds(self.rounds),
            "question": question,
        }

    @staticmethod
    def flatten_rounds(rounds: list[list[BaseMessage]]) -> list[BaseMessage]:
        return [m for turn in rounds for m in turn]

    def __repr__(self) -> str:
        return (
            f"MyAgent(friend_name={self.friend_name!r}, "
            f"rounds={len(self.rounds)}, "
            f"memory_len={len(self.memory)}, "
            f"tools={list(self.tool_map)!r})"
        )

    def __str__(self) -> str:
        table = []
        for msg in self.prompt.invoke(self._chat_payload("")).to_messages():
            table.append([msg.type, msg.content])
        return tabulate(table, headers=["角色", "内容"])


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
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
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
def get_user_city() -> str:
    """返回演示用的用户所在城市。

    Returns:
        城市名称字符串。
    """
    return "广州"



def build_agent() -> MyAgent:
    """根据环境变量构造 ``MyAgent`` 实例。

    Returns:
        配置完成的 ``MyAgent``。

    Raises:
        ValueError: 未设置 ``DEEPSEEK_API_KEY`` 环境变量。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return MyAgent(
        api_key=api_key,
        friend_name="小云",
        tools=[get_weather, get_user_city],
        keep_raw_rounds=3,
        buffer_rounds=2,
    )


@asynccontextmanager
async def _lifespan(application: FastAPI):
    application.state.agent = build_agent()
    application.state.agent_lock = asyncio.Lock()
    yield


app = FastAPI(title="MyAgent API", lifespan=_lifespan)


@app.post("/chat/invoke")
async def http_ainvoke(question: str = Body(..., embed=True, min_length=1)):
    """非流式对话：等待整轮结束后返回完整文本。"""
    agent: MyAgent = app.state.agent
    lock: asyncio.Lock = app.state.agent_lock
    async with lock:
        content = await agent.ainvoke(question)
    print(repr(agent))
    print(str(agent))
    return {"content": content}


@app.post("/chat/stream")
async def http_astream(question: str = Body(..., embed=True, min_length=1)):
    """流式对话：以 SSE（``text/event-stream``）逐块返回文本。"""
    agent: MyAgent = app.state.agent
    lock: asyncio.Lock = app.state.agent_lock

    async def sse_chunks() -> AsyncIterator[str]:
        async with lock:
            async for chunk in agent.astream(question):
                payload = json.dumps(
                    {"content": chunk}, ensure_ascii=False
                )
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
    print(repr(agent))
    print(str(agent))
    return StreamingResponse(sse_chunks(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
