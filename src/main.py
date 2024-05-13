# Streamlit 应用程序界面
import streamlit as st
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# api_key: 609356d25d7f1939ab974b04b4e803a1.EjZO8ylvTqlLxWve
def generate_response(input_text,openai_api_key):
    chat = ChatZhipuAI(model="glm-3-turbo",temperature=0.5, zhipuai_api_key=openai_api_key)
    messages = [
        # AIMessage(content="Hi."),
        # SystemMessage(content="Your role is a poet."),
        HumanMessage(content=input_text),
    ]   
    resp = chat.invoke(messages)
    st.info(resp.content)
def main():
    st.title('🦜🔗 动手学大模型应用开发')
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        # 调用 respond 函数获取回答
        answer = generate_response(prompt, openai_api_key)
        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   
main()