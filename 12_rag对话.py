import os
from pathlib import Path

from tabulate import tabulate

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


MAX_ROUNDS = 3
CHROMA_DIR = Path("data/chroma_db")  # 向量库持久化目录
COLLECTION_NAME = "employee_handbook"  # 集合名
K = 3  # 检索结果数量

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", (
            "你是公司员工助手，负责回答员工关于公司规章制度的问题。\n\n"
            "【参考资料】\n"
            "{context}\n\n"
            "回答规则：\n"
            "1. 只根据【参考资料】中的内容回答\n"
            "2. 如果资料中没有相关信息，直接说「手册中未找到相关信息」\n"
            "3. 不要编造或推测资料中没有的内容\n"
            "4. 回答尽量简洁，引用具体条款时注明来源"
        )),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)

embeddings = HuggingFaceEmbeddings(
    model_name="text-embedding-3-small",
    model_kwargs={"device": "cpu"},
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": K})

chain = prompt | llm

history = []

while True:
    user_input = input("> ").strip()
    if not user_input:
        continue
    if user_input == "exit":
        break

    context = retriever.invoke(user_input)

    payload = {
        "role": "小云",
        "history": history,
        "user_input": user_input,
        "context": context,
    }
    system_msg_content = prompt.invoke(payload).to_messages()[0].content
    # 等价于：
    # for chunk in llm.stream(prompt.invoke(history)):
    ai_msg: AIMessageChunk | None = None
    for chunk in chain.stream(payload):
        ai_msg = chunk if ai_msg is None else ai_msg + chunk
        print(chunk.content, end="", flush=True)
    print()

    history.append(HumanMessage(content=user_input))
    history.append(ai_msg)
    history = history[-(MAX_ROUNDS * 2):]

    table: list[list] = [
        ["system", system_msg_content]
    ]
    for msg in history:
        table.append([msg.type, msg.content])
    print(tabulate(table, headers=["角色", "内容"], tablefmt="grid"))
