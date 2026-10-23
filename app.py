import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.agents import create_pandas_dataframe_agent

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

                # Smart routing with expanded keywords for real-time lookups
                query_lower = prompt.lower()
                search_keywords = [
                    "search", "latest", "news", "who is", "what is", "where is", "when is",
                    "weather", "current", "minister", "president", "prime minister",
                    "cm", "pm", "chief minister", "governor", "ceo", "head"
                ]

                if any(kw in query_lower.split() or kw in query_lower for kw in search_keywords):
                    response = search.run(prompt)
                else:
                    try:
                        response = pandas_agent.run(prompt)
                    except Exception:
                        # Fallback to direct LLM response if pandas agent fails/parses incorrectly
                        fallback_res = llm.invoke(prompt)
                        response = fallback_res.content if hasattr(fallback_res, 'content') else str(fallback_res)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})