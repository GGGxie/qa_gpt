import os,sys
# 获取当前文件的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录的路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "src"))
print(parent_dir)
# 将上层目录添加到Python路径
sys.path.append(parent_dir)

from zhipu_embedding import ZhipuAIEmbeddings
from langchain_chroma import Chroma
persist_directory = 'data/vector_db/chroma'
db3 = Chroma(persist_directory=persist_directory, embedding_function=ZhipuAIEmbeddings())
docs = db3.similarity_search('LangSmith')
print(docs)