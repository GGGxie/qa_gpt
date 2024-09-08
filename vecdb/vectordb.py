from langchain_chroma import Chroma
from .loader import vec_path, embedding_func

# vectordb = Chroma(embedding_function=ZhipuAIEmbeddings(), persist_directory=persist_directory)
# 设置代理

vectordb = Chroma(
    embedding_function=embedding_func,
    persist_directory=vec_path,
)


# 加载向量数据库
def get_vectordb():
    return vectordb
