import requests
import os
from langgraph.graph import StateGraph
from twilio.rest import Client
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, BaseMessage
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain.schema import Document


class FeishuRoot:
    def __init__(self):
        self.webHood = os.getenv("WEBHOOD")

    def send_message(self, message):
        url = f"https://open.feishu.cn/open-apis/bot/v2/hook/{self.webHood}"
        headers = {"Content-Type": "application/json"}
        data = {"msg_type": "text", "content": {"text": message}}
        response = requests.post(url, headers=headers, json=data).json()
        if response["code"] == 0:
            return "发送成功"
        else:
            return "发送失败"


class GeoAPI:
    def __init__(self):
        self.key = os.getenv("WEATHER_KEY")

    def lookup(self, city):
        # url = (
        #     f"https://geoapi.qweather.com/v2/city/lookup?location={city}&key={self.key}"
        # )
        # response = requests.get(url).json()
        # if response["code"] == "200":
        #     id = response["location"][0]["id"]
        #     return id
        # else:
        #     return None
        # 深圳
        return "101280601"

    def get_weather(self, city, days="3d"):
        location_id = self.lookup(city)
        if location_id is None:
            return None
        url = f"https://devapi.qweather.com/v7/weather/{days}?location={location_id}&key={self.key}"
        response = requests.get(url).json()
        if response["code"] == "200":
            data = response["daily"]
            return data
        else:
            return None


# 定义获取天气的函数
@tool
def get_weather(location: str):
    """Call to get the current weather."""
    geo = GeoAPI()
    data = geo.get_weather(location)
    return data


# Twilio 发送短信的功能
@tool
def send_message(message: str):
    """Send message to feishu root"""
    feishuclient = FeishuRoot()
    ret = feishuclient.send_message(message)
    return ret


# 定义 GraphState，用于存储运行时状态
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class MyAgent:
    def __init__(self, llm, retriever):
        # 初始化工作流
        self.retriever = retriever
        tools = [get_weather, send_message]
        self.tool_node = ToolNode(tools)
        self.chain = llm.bind_tools(tools)
        self.init_graph()

    def generate(self, state):
        """根据筛选的文档生成回复"""
        messages = state["messages"]
        response = self.chain.invoke(messages)

        if isinstance(response, AIMessage):
            state["messages"].append(response)
        else:
            raise TypeError(f"Expected AIMessage but got {type(response)}")

        return state

    def should_continue(self, state: GraphState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    def init_graph(self):
        workflow = StateGraph(GraphState)

        # 添加节点
        workflow.add_node("agent", self.generate)
        workflow.add_node("tools", self.tool_node)

        # 设置节点间的连接
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self.should_continue)
        workflow.add_edge("tools", "agent")

        self.app = workflow.compile()

    # 示例调用
    def run(self, question):
        state = GraphState(messages=[HumanMessage(content=question)])
        final_state = self.app.invoke(state)
        return final_state["messages"][-1].content


# 运行
if __name__ == "__main__":
    agent = MyAgent()
    agent.run("FoShan", "+1234567890")
