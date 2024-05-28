from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from zhipu_embedding import ZhipuAIEmbeddings
from langchain_chroma import Chroma
persist_directory = 'data/vector_db/chroma'
db = Chroma.from_documents(documents=all_splits, embedding=ZhipuAIEmbeddings(), persist_directory=persist_directory)

