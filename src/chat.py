from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain import hub
from langchain_core.messages import AnyMessage
from typing import Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class AutoChat:
    def __init__(self, llm, retriver, temperature=0.9, max_tokens=1024):
        self.llm = llm
        self.prompt = PromptTemplate.from_file(
            "prompts/chat/chat.txt", ["context", "question"]
        )
        print(self.prompt)
        self.retriver = retriver

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Return the result as a JSON object with a key 'binary_score' and a value of 'yes' or 'no'to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )
        parser = PydanticOutputParser(pydantic_object=GradeDocuments)
        self.retrieval_grader = grade_prompt | self.llm | parser
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.web_search_tool = TavilySearchResults(k=3)
        self.rag_chain = self.prompt | self.llm
        self.init_graph()

    def init_graph(self):
        workflow = StateGraph(GraphState)
        # 添加节点
        workflow.add_node("generate", self.generate)
        # workflow.add_node("web_search", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        # 添加边
        # workflow.add_conditional_edges("generate", self.should_continue)
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("generate", END)
        # 设置入口点
        workflow.set_entry_point("retrieve")
        # 使用 MemorySaver 持久化存储
        self.app = workflow.compile(checkpointer=memory)

    def grade_documents(self, state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        state["documents"] = filtered_docs
        return state

    def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        # RAG generation
        response = self.rag_chain.invoke(
            # message
            {"context": documents, "question": question}
        )
        # state["messages"].append(response)
        if isinstance(response, AIMessage):
            generation = response.content
        else:
            raise TypeError(f"Expected AIMessage but got {type(response)}")
        state["generation"] = generation
        return state

    def _load_prompts(self, file_path):
        prompts = {}
        with open(file_path, "r") as file:
            content = file.read()
            sections = content.split("\n\n")
            for section in sections:
                lines = section.split("\n", 1)
                if len(lines) == 2:
                    key, prompt = lines
                    prompts[key.rstrip(":")] = prompt.strip()
        return prompts

    def web_search(self, state):
        state["search_flag"] = True
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
        print("---RETRIEVE---")
        question = state["question"]
        # Retrieval
        try:
            documents = self.retriver.invoke(question)
        except Exception as e:
            print(e)
        print(documents)
        state["documents"] = documents
        return state

    def should_continue(self, state) -> str:
        last_message = state["generation"]
        if (
            "I don't know" in last_message or "I cannot" in last_message
        ) and not state.get("search_flag", False):
            return "web_search"
        return END

    def run(self, task: str, thread_id="2") -> str:
        state = GraphState(
            messages=HumanMessage(content=task),
            question=task,
            generation="",
            documents=[],
            search_flag=False,
        )
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.app.invoke(state, config)
        print(state)
        return final_state["generation"]


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    generation: str
    documents: List[str]
    search_flag: bool


memory = MemorySaver()
