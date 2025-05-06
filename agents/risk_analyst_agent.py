from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
import pandas as pd

# --- Load claim data ---
def load_claim_summary():
    df = pd.read_csv("data/classification_data.csv")
    summary = df.describe(include='all').to_string()
    return summary

# --- Define tool for the agent ---
def risk_summary_tool():
    return Tool(
        name="Risk Summary Generator",
        func=lambda _: load_claim_summary(),
        description="Returns statistical summary of historical insurance claim data for risk assessment"
    )

# --- Set up LangChain LLM agent ---
def create_risk_analyst_agent():
    llm = ChatOpenAI(temperature=0.3, model="gpt-4")
    tools = [risk_summary_tool()]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    return agent

# --- Example: Agent interaction ---
if __name__ == "__main__":
    agent = create_risk_analyst_agent()
    result = agent.invoke({"input": "Give me a risk summary of recent insurance claims and trends"})
    print("\n--- Agent Response ---\n")
    print(result.get("output"))
