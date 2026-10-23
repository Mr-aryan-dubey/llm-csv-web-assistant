import os
import pandas as pd
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# 1. Load Environment Variables from .env
load_dotenv()

# 2. Load the Local CSV Dataset
CSV_FILE_PATH = "data/dataset.csv"

if not os.path.exists(CSV_FILE_PATH):
    print(f"Error: Could not find '{CSV_FILE_PATH}'. Please place your CSV file in the 'data/' folder.")
    exit()

df = pd.read_csv(CSV_FILE_PATH)

# 3. Initialize LLM for the Pandas DataFrame Agent
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

csv_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    allow_dangerous_code=True
)


@tool
def query_csv_data(query: str) -> str:
    """Useful for answering questions about the local CSV dataset, performing calculations, filtering data, or summarizing rows/columns."""
    try:
        response = csv_agent.invoke({"input": query})
        return str(response.get("output", response))
    except Exception as e:
        return f"Error executing CSV query: {str(e)}"


# 4. Define Tool 2: Web Search Tool
search_engine = DuckDuckGoSearchRun()


@tool
def search_web(query: str) -> str:
    """Useful for searching Google/the web for current facts, external information, or topics not present in the CSV file."""
    try:
        return search_engine.run(query)
    except Exception as e:
        return f"Error performing web search: {str(e)}"


# 5. Combine Tools and Initialize Modern Agent
tools = [query_csv_data, search_web]

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=tools,
    system_prompt=(
        "You are an intelligent AI Assistant capable of querying local CSV data as well as searching the web. "
        "Analyze the user's input carefully: "
        "- If the question is about the provided CSV data (e.g., stats, columns, rows, counts), use the `query_csv_data` tool. "
        "- If the question requires general knowledge, current events, or external facts, use the `search_web` tool. "
        "- Synthesize clear, accurate responses for the user."
    )
)

# 6. Run Interactive Loop
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Multi-Tool AI Assistant (CSV Data + Web Search) Ready!")
    print("=" * 60)

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting application...")
            break

        if not user_input:
            continue

        print("\nProcessing request...")
        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print(f"\nAI: {response['messages'][-1].content}")