# Streamlit 应用程序界面
import streamlit as st
import answer,os

# api_key: 609356d25d7f1939ab974b04b4e803a1.EjZO8ylvTqlLxWve

def main():
    st.title('🦜🔗 动手学大模型应用开发')
    zhipu_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    zhipu_api_key= '609356d25d7f1939ab974b04b4e803a1.EjZO8ylvTqlLxWve'
    os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key
    # 添加一个选择按钮来选择不同的模型
    # selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])
    
    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 调用 respond 函数获取回答
        st.session_state.messages.append({"role": "user", "text": prompt})
        if selected_method == 'None':
            answer = answer.generate_response(prompt)
        elif selected_method=='qa_chain':
            answer = answer.get_qa_chain(prompt)
        elif selected_method =='chat_qa_chain':
            answer = answer.get_chat_qa_chain(prompt) 
        
        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if  ["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   
main()