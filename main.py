from dotenv import load_dotenv

load_dotenv()

from  langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="search for job postings for data engineer in quebec city using langchain on linkedin and list their details")})
    print(f"Agent result: {result}")    


if __name__ == "__main__":
    main()
