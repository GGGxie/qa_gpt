# Streamlit 应用程序界面
# 加载环境变量
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
import os, sys


# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(".")

import streamlit as st
from src import chat, agent
from langchain_openai import ChatOpenAI
from vecdb import vectordb, loader


def main():
    # 设置代理
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    # 语言模型
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0,
        # model_kwargs={"seed": 42},
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    auto_chat = chat.AutoChat(
        llm=llm,
        retriever=vectordb.get_vectordb().as_retriever(),
        # tools=tools,
        # work_dir="./data",
        # main_prompt_file="./prompts/main/main.txt",
        # max_thought_steps=20,
    )
    my_agent = agent.MyAgent(
        llm=llm,
        retriever=vectordb.get_vectordb().as_retriever(),
    )

    st.title("一、知识库文档")

    # 上传文件
    uploaded_file = st.file_uploader("上传 PDF 或 XLSX 文件", type=["pdf", "xlsx"])

    if uploaded_file is not None:
        loader.handle_uploaded_file(uploaded_file)

    st.title("二、个人助手")
    # zhipu_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    # 添加一个选择按钮来选择不同的模型
    # selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        # ["None", "qa_chain", "chat_qa_chain"],
        ["gpt4-o", "agent"],
        captions=[
            "gpt4-o大模型",
            "私人助手",
            # "不使用检索问答的普通模式",
            # "不带历史记录的检索问答模式",
            # "带历史记录的检索问答模式",
        ],
    )

    # 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if task := st.chat_input("Say something"):
        # 调用 respond 函数获取回答
        st.session_state.messages.append({"role": "user", "text": task})
        if selected_method == "gpt4-o":
            answer = auto_chat.run(task)
        elif selected_method == "agent":
            answer = my_agent.run(task)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if ["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])


if __name__ == "__main__":
    # print(config)
    main()
