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
    """When was Zup founded?"""
    return "Zup was founded in 2011."

@tool
def calculate_square_root(x: int) -> float:
    """Calculate the square root of a number"""
    return x**0.5

class TestModel(BaseModel):
    """Answers when Zup was founded"""
    query: str = Field(description="The query to answer")

def answer_query(self, query: str) -> str:
    return {"response": "Zup was founded in 2011."}


get_zup_query_tool = Tool(
    name="get_zup_query",
    func=answer_query,
    description="Answers when Zup was founded",
    args_schema=TestModel
)

def main():
    
    # chain = prompt | model
    # result = chain.invoke({"input": "When was Zup founded?"})
    tools = [when_was_zup_founded, calculate_square_root]
    # tools = [get_zup_query_tool]
    # prompt = ChatPromptTemplate.from_template(
    #     "[INS]You're an expert in the tech field. Anser the question: {input}[/INS] \n{agent_scratchpad}", tools=tools
    # )
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": "What's the square root of 25"})
    print(result)

if __name__ == "__main__":
    main()

