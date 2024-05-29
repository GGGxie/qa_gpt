from zhipu_embedding import ZhipuAIEmbeddings

from langchain_chroma import Chroma

### Statefully manage chat history ###
store = {}
persist_directory = 'data/vector_db/chroma'
vectordb = Chroma(embedding_function=ZhipuAIEmbeddings(),persist_directory=persist_directory)

# 加载向量数据库
def get_vectordb():
    return vectordb