from langchain_community.chat_models import ChatZhipuAI
from langchain.memory import ConversationBufferMemory
import zhipu_embedding
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain_core.messages import HumanMessage
class VectorDB():
    vectordb: Chroma
    def __init__(self,ZhipuAIEmbeddings,path,documents):
        self.vectordb =Chroma.from_documents(
            embedding=ZhipuAIEmbeddings,
            persist_directory=path,
            documents=documents
        )
    def persist(self):
        self.vectordb.persist()# 允许我们将persist_directory目录保存到磁盘上

# 加载向量数据库
def get_vectordb():
    # 定义 Embeddings
    embedding = zhipu_embedding.ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = 'data/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
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
    vectordb = get_vectordb()
    llm = ChatZhipuAI(model_name = "gpt-3.5-turbo", temperature = 0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa.invoke({"question": question})
    return result['answer']