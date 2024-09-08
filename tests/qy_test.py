from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

chatLLM = ChatTongyi(
    streaming=True,
)
print(chatLLM.dashscope_api_key)
# res = chatLLM.stream([HumanMessage(content="hi")], streaming=True)
# for r in res:
#     print("chat resp:", r)