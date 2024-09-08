from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

import os
from langchain_community.document_loaders import PyMuPDFLoader

# 加载环境变量

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredExcelLoader
import traceback

# 设置路径
data_path = "data"
vec_path = data_path + "/vector_db/chroma"

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")


embedding_func = OpenAIEmbeddings(
    openai_api_key=api_key,
    openai_api_base=base_url,
)


# 加载Excel文件
def process_excel(file_path):
    loader = UnstructuredExcelLoader(file_path)
    documents = loader.load()
    return documents


# 加载PDF文件
def process_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


# 处理并向量化文件
def process_file(file_path):
    if file_path.endswith(".xlsx"):
        documents = process_excel(file_path)
    elif file_path.endswith(".pdf"):
        documents = process_pdf(file_path)
    else:
        raise ValueError("Unsupported file type")
    return documents


if __name__ == "__main__":
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # 遍历data文件夹中的所有文件
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".xlsx") or file.endswith(".pdf"):
                try:
                    documents = process_file(file_path)
                    docs = text_splitter.split_documents(documents)
                    db = Chroma.from_documents(
                        docs, embedding_func, persist_directory=vec_path
                    )
                except Exception as e:
                    # 获取异常类型
                    error_type = type(e).__name__

                    # 输出完整的堆栈跟踪
                    tb_str = traceback.format_exc()

                    # 打印文件路径、异常类型和详细的堆栈跟踪信息
                    print(f"Error storing file {file_path}: {error_type}")
                    print(f"Exception message: {str(e)}")
                    print("Stack trace:")
                    print(tb_str)
