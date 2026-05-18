import os
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 同步
# def main():
#     result = llm.invoke("你好，介绍一下你自己")
#     print(result.content)

# 异步
# async def main():
#     result = await llm.ainvoke("你好，介绍一下你自己")
#     print(result.content)

# 同步流式
# def main():
#     result = llm.stream("你好，介绍一下你自己")
#     for chunk in result:
#         print(chunk.content, end="", flush=True)
#     print()

# 异步流式


async def main():
    result = llm.astream("你好，介绍一下你自己")
    async for chunk in result:
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
