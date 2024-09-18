from typing import List, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph, add_messages
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults


# 定义评分结果的 Pydantic 模型
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# 定义 GraphState，用于存储运行时状态
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    generation: str
    documents: List[Document]
    search_flag: bool


# 主类，包含文档检索、评分、生成对话等功能
class AutoChat:
    def __init__(self, llm, retriever, temperature=0.9, max_tokens=1024):
        self.llm = llm
        self.retriever = retriever
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.web_search_tool = TavilySearchResults(k=3)

        # 加载提示模板
        self.prompt = PromptTemplate.from_file(
            "prompts/chat/chat.txt", ["context", "question"]
        )

        # 定义评分提示和解析器
        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Return the result as a JSON object with a key 'binary_score' and a value of 'yes' or 'no'to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Retrieved document: {document}\n\nUser question: {question}",
                ),
            ]
        )
        parser = PydanticOutputParser(pydantic_object=GradeDocuments)
        self.retrieval_grader = grade_prompt | self.llm | parser

        # 初始化生成链
        self.rag_chain = self.prompt | self.llm

        # 初始化流程图
        self.init_graph()

    def init_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("generate", self.generate)
        workflow.add_edge("retrieve", "web_search")
        workflow.add_edge("web_search", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("generate", END)
        workflow.set_entry_point("retrieve")

        # 使用 MemorySaver 来持久化存储状态
        self.app = workflow.compile(checkpointer=memory)

    def grade_documents(self, state):
        """筛选与用户问题相关的文档"""
        question = state["question"]
        documents = state["documents"]
        relevant_docs = []
        if documents is not None:
            for doc in documents:
                score = self.retrieval_grader.invoke(
                    {"question": question, "document": doc.page_content}
                )
                if score.binary_score == "yes":
                    relevant_docs.append(doc)

        state["documents"] = relevant_docs
        return state

    def generate(self, state):
        """根据筛选的文档生成回复"""
        question = state["question"]
        documents = state["documents"]
        response = self.rag_chain.invoke({"context": documents, "question": question})

        if isinstance(response, AIMessage):
            state["generation"] = response.content
        else:
            raise TypeError(f"Expected AIMessage but got {type(response)}")

        return state

    def web_search(self, state):
        """执行网络搜索"""
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        if isinstance(docs, str):
            docs = [docs]

        web_results = "\n".join(
            [
                d["content"] if isinstance(d, dict) and "content" in d else d
                for d in docs
            ]
        )
        web_results = Document(page_content=web_results)

        state["documents"].append(web_results)
        return state

    def retrieve(self, state):
        """检索相关文档"""
        question = state["question"]
        try:
            documents = self.retriever.invoke(question)
            state["documents"] = documents
        except Exception as e:
            print(f"Retrieval failed: {e}")
            state["documents"] = []

        return state

    def run(self, task: str, thread_id="2") -> str:
        """执行任务并返回生成的回复"""
        state = GraphState(
            messages=[HumanMessage(content=task)],
            question=task,
            generation="",
            documents=[],
            search_flag=False,
        )
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.app.invoke(state, config)
        return final_state["generation"]


# MemorySaver 用于存储历史
memory = MemorySaver()
