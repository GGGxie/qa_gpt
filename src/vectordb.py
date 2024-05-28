from zhipu_embedding import ZhipuAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatZhipuAI
from langchain.chains import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage
from chromadb.config import Settings

### Statefully manage chat history ###
store = {}
# class VectorDB():
#     vectordb: Chroma
#     def __init__(self,embedding, persist_directory):
#         self.vectordb =Chroma(
#             embedding_function=embedding,
#             persist_directory=persist_directory  
#         )
persist_directory = 'data/vector_db/chroma'
vectordb = Chroma(embedding_function=ZhipuAIEmbeddings(),persist_directory=persist_directory)

# 加载向量数据库
def get_vectordb():
    return vectordb

def generate_response(input_text):
    chat = ChatZhipuAI(model="glm-3-turbo",temperature=0.5)
    messages = [
        # AIMessage(content="Hi."),
        # SystemMessage(content="Your role is a poet."),
        HumanMessage(content=input_text),
    ]   
    resp = chat.invoke(messages)
    return resp.content

#不带历史记录的问答链
def get_qa_chain(question:str):
    vectordb = get_vectordb()
    llm = ChatZhipuAI(model="glm-3-turbo",temperature=0)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

#带有历史记录的问答链
def get_chat_qa_chain(question:str):
    llm = ChatZhipuAI(model="glm-3-turbo",temperature=0)
    vectordbt = get_vectordb()
    retriever=vectordbt.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": "abc123"}}
    )["answer"]