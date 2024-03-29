from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import tool, Tool
from langchain_community.llms.llamafile import Llamafile
# from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

model = Llamafile(streaming=True, temperature=0, n_predict=300)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)


@tool
def when_was_zup_founded():
    """Tells the date Zup was founded"""
    return "Zup was founded in 2011."


@tool
def calculate_square_root(x: int) -> float:
    """Calculates the square root of a number"""
    return x**0.5


def main():
    tools = [when_was_zup_founded, calculate_square_root]
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": "When was Zup founded?"})
    print(result)

if __name__ == "__main__":
    main()

