from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_community.chat_models import ChatZhipuAI
import os

# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY=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
# export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# export LANGCHAIN_PROJECT="test"

zhipu_api_key= '609356d25d7f1939ab974b04b4e803a1.EjZO8ylvTqlLxWve'
os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key
llm = ChatZhipuAI(model="glm-3-turbo",temperature=0)
# Auto-trace LLM calls in-context

@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content

pipeline("Hello, world!")
# Out:  Hello there! How can I assist you today?