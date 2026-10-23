import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Retrieve API Key from Streamlit Secrets or local environment
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Page Configuration
st.set_page_config(
    page_title="LLM CSV & Web Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-Tool AI Assistant")
st.markdown("Query your local CSV data or search the live web instantly using **Groq**.")

# Sidebar for file upload only
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded `{uploaded_file.name}` successfully!")
        with st.expander("Preview Dataset"):
            st.dataframe(df.head())
    else:
        # Fallback to default sample dataframe if none uploaded
        df = pd.DataFrame({
            "Sample_Column": ["No file uploaded yet. Upload a CSV above or ask general web queries."],
            "Value": [0]
        })

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Box
if prompt := st.chat_input("Ask about your CSV or search the web..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching..."):
            try:
                # Setup LLM with the active Groq model
                llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.1)

                # Setup Tools (Pandas Agent + Web Search)
                search = DuckDuckGoSearchRun()

                # Create pandas dataframe agent
                pandas_agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=False,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )

                # System persona defining creator identity
                system_persona = (
                    "You are a helpful Multi-Tool AI Assistant built and created by Mr. Aryan Dubey. "
                    "If anyone asks who created you, who built you, or who your developer is, "
                    "you must state clearly and proudly that you were created and built by Mr. Aryan Dubey."
                )

                query_lower = prompt.lower()
                search_keywords = [
                    "search", "latest", "news", "who is", "what is", "where is", "when is",
                    "weather", "current", "minister", "president", "prime minister",
                    "cm", "pm", "chief minister", "governor", "ceo", "head"
                ]

                # 1. Creator Identity Query Routing
                if any(k in query_lower for k in
                       ["created you", "built you", "your creator", "who made you", "who designed you",
                        "who developed you"]):
                    response = "I was created and built by **Mr. Aryan Dubey**!"

                # 2. Live Web Search Query Routing
                elif any(kw in query_lower.split() or kw in query_lower for kw in search_keywords):
                    search_results = search.run(prompt)

                    synthesis_prompt = (
                        f"Based strictly on the following web search results, provide a direct and accurate answer to: '{prompt}'.\n"
                        f"CRITICAL INSTRUCTION: Pay close attention to recent dates and leadership changes in 2026 (such as Samrat Choudhary serving as Chief Minister of Bihar). "
                        f"Do not default to former officeholders if a new person has taken office.\n\n"
                        f"Search Results:\n{search_results}"
                    )
                    messages = [
                        SystemMessage(content=system_persona),
                        HumanMessage(content=synthesis_prompt)
                    ]
                    fallback_res = llm.invoke(messages)
                    response = fallback_res.content if hasattr(fallback_res, 'content') else str(fallback_res)

                # 3. CSV Analysis & Fallback Query Routing
                else:
                    try:
                        response = pandas_agent.run(prompt)
                    except Exception:
                        messages = [
                            SystemMessage(content=system_persona),
                            HumanMessage(content=prompt)
                        ]
                        fallback_res = llm.invoke(messages)
                        response = fallback_res.content if hasattr(fallback_res, 'content') else str(fallback_res)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})