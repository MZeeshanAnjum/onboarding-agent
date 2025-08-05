from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
from langchain_core.messages import HumanMessage, AIMessage
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=OPENAI_API_KEY  
)

if __name__ == "__main__":
    print(llm.invoke([HumanMessage(content="What tools do I need to complete my task?")]))