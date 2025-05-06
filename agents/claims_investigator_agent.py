from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import pandas as pd
import numpy as np

# --- Simulated policy limit checker ---
def check_policy_limits(claim_id: str) -> str:
    df = pd.read_csv("data/classification_data.csv")
    df["claim_id"] = [f"CLAIM{str(i).zfill(4)}" for i in range(1, len(df) + 1)]
    df["policy_limit"] = df["claims_count"] * 800 + 5000
    df["amount_paid"] = df["claims_count"] * 1000 + np.random.normal(0, 500, size=len(df)).clip(min=0)
    record = df[df["claim_id"] == claim_id.upper()]
    if record.empty:
        return f"Claim ID {claim_id} not found in the database."
    row = record.iloc[0]
    if row["amount_paid"] > row["policy_limit"]:
        return (f"\U0001F6A8 Claim {claim_id} exceeds policy limit.\n"
                f"\tAmount Paid: ${row['amount_paid']:.2f}\n"
                f"\tPolicy Limit: ${row['policy_limit']:.2f}\n"
                f"\tAction: Flag for audit and notify underwriting team.")
    else:
        return (f"\u2705 Claim {claim_id} is within policy limit.\n"
                f"\tAmount Paid: ${row['amount_paid']:.2f}\n"
                f"\tPolicy Limit: ${row['policy_limit']:.2f}\n"
                f"\tAction: No anomaly detected.")

policy_check_tool = Tool(
    name="Policy Limit Validator",
    func=check_policy_limits,
    description="Check if an insurance claim exceeds its policy limit by supplying the claim ID."
)

def create_claims_investigator_agent():
    llm = ChatOpenAI(temperature=0.2, model="gpt-4")
    tools = [policy_check_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    return agent

if __name__ == "__main__":
    agent = create_claims_investigator_agent()
    test_claims = ["CLAIM0001", "CLAIM0015", "CLAIM0090"]
    prompt = ("For the following claims: " + ", ".join(test_claims) +
              ", check if they exceed their policy limits and summarize findings with next steps.")
    result = agent.invoke({"input": prompt})
    print("\n--- Agent Report ---\n")
    print(result.get("output"))
