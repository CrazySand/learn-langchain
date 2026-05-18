import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek

# 折叠后 history 中保留的原文轮数（每轮 = Human + AI，共 2 条消息）
KEEP_RAW_ROUNDS = 3
# 触发摘要前允许多攒的额外轮数；软上限 = (KEEP_RAW_ROUNDS + BUFFER_ROUNDS) * 2 条消息
BUFFER_ROUNDS = 2

llm = ChatDeepSeek(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是{role}，我的好朋友。\n\n"
            "【较早对话摘要】（仅作背景；若写「尚无」表示暂无）\n"
            "{memory}\n\n"
            "下列 history 为人类与助手最近几轮原文，请结合摘要与原文回答；不要编造摘要与原文中均未出现的事实。",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# LCEL：先渲染模板，再把 messages 交给同一套 llm（与非链式的 prompt.invoke + llm.stream 等价）
chain = prompt | llm

memory = ""

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


def merge_into_summary(llm: ChatDeepSeek, prev_summary: str, fold_messages: list) -> str:
    """把将被裁掉的对话片段合并进滚动摘要（额外一次模型调用）。

    Args:
        llm: 用于生成摘要的聊天模型实例。
        prev_summary: 当前的滚动摘要文本；若尚无则为空字符串。
        fold_messages: 本轮需要从 history_raw 移除、并入摘要的消息列表。

    Returns:
        合并后的新摘要字符串。
    """
    prompt_text = (
        "你是一个对话压缩助手。请将「已有摘要」与「新片段」合并成一份更短的中文要点列表"
        "（可编号），严格基于原文，不要编造未出现的信息。\n"
        "务必保留：数字、专名、用户明确约束与待办事项。\n\n"
        f"【已有摘要】\n{prev_summary.strip() or '（尚无）'}\n\n"
        f"【新片段】\n{transcript(fold_messages)}\n\n"
        "【合并后的摘要】"
    )
    out = llm.invoke(prompt_text)
    return out.content


while True:
    user_input = input("> ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    payload = {
        "role": "小明",
        "memory": memory.strip() or "（尚无）",
        "history": history,
        "question": user_input,
    }

    debug(prompt.invoke(payload).to_messages())

    pieces: list[str] = []
    for chunk in chain.stream(payload):
        if chunk.content:
            pieces.append(chunk.content)
            print(chunk.content, end="", flush=True)
    print()

    # 非流式等价写法：ai_msg = chain.invoke(payload)

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content="".join(pieces)))

    max_keep_msgs = KEEP_RAW_ROUNDS * 2
    soft_max_msgs = (KEEP_RAW_ROUNDS + BUFFER_ROUNDS) * 2
    if len(history) > soft_max_msgs:
        # fold 代表将被裁掉的对话片段，将它合并进滚动摘要
        fold = history[:-max_keep_msgs]
        # 更新 history，只保留最近的 KEEP_RAW_ROUNDS 轮对话
        history = history[-max_keep_msgs:]
        # 将 fold 合并进滚动摘要，更新记忆
        memory = merge_into_summary(llm, memory, fold)
