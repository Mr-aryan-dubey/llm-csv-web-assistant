# 🤖 Multi-Tool AI Assistant (CSV & Web Search)

An interactive, multi-tool AI assistant built using **Streamlit**, **LangChain**, and **Groq (`openai/gpt-oss-120b`)**. This web application enables natural language data analysis on uploaded CSV files alongside live web searching via DuckDuckGo, complete with smart query routing and graceful error fallback.

---

## ✨ Features

- **📊 CSV Data Analysis**: Upload any CSV dataset to run complex analytical queries using LangChain's Pandas DataFrame Agent.
- **🌐 Real-Time Web Search**: Automatically routes queries requiring current events or general facts to DuckDuckGo Search.
- **🛡️ Error-Resistant Routing**: Includes automatic fallback mechanisms to handle agent parsing errors without breaking the user experience.
- **🔒 Secure Configuration**: Uses Streamlit Secrets and `.env` files for seamless API key management without exposing credentials in code.
- **💬 Interactive Chat Interface**: Features persistent chat history and continuous conversation state.

---

## 🛠️ Tech Stack

- **Frontend & UI**: Streamlit
- **LLM Engine**: Groq API (`openai/gpt-oss-120b`)
- **Framework**: LangChain (`langchain-groq`, `langchain-experimental`, `langchain-community`)
- **Data Handling**: Pandas, Tabulate
- **Search Tool**: DuckDuckGo Search (`duckduckgo-search`, `ddgs`)
- **Environment Management**: `python-dotenv`

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ installed on your system.
- A free **Groq API Key** from [console.groq.com](https://console.groq.com).

### 2. Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Mr-aryan-dubey/llm-csv-web-assistant.git](https://github.com/Mr-aryan-dubey/llm-csv-web-assistant.git)
   cd llm-csv-web-assistant
   

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Configure environment variables:
Create a .env file in the root project directory:
GROQ_API_KEY=your_groq_api_key_here

# Run the Streamlit application
python -m streamlit run app.py

