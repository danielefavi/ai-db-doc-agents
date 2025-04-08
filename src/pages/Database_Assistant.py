import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
import streamlit as st

from libs.sql_query_assistant import SQLQueryAssistant
from libs.ui_components import render_ollama_model_selector


DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL")

DB_DETAILS = {
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASS"),
    'database': os.getenv("DB_NAME")
}

if "ai_assistant" not in st.session_state:
    llm_model = ChatOllama(model=DEFAULT_LLM_MODEL)
    st.session_state.ai_assistant = SQLQueryAssistant(db_config=DB_DETAILS, llm_model=llm_model)

# Ensure DB details are present (add error handling if needed)
if not all(DB_DETAILS.values()):
    st.error("Database connection details missing in environment variables (DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME)")
    st.stop() # Stop execution if DB details are missing


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm the AI Database assistant, how can I help you?"),
    ]



st.title("Chat to Database")



with st.sidebar:
    selected_llm = render_ollama_model_selector(
        default_model_name=DEFAULT_LLM_MODEL
    )
    llm_model = ChatOllama(model=selected_llm)
    st.session_state.ai_assistant = SQLQueryAssistant(db_config=DB_DETAILS, llm_model=llm_model)

    st.markdown("---") 

    st.code(f"""
        DEFAULT_LLM_MODEL="{DEFAULT_LLM_MODEL}"
        SELECTED_LLM="{selected_llm}"
    """)

    st.markdown("---") 

    st.markdown(
        """
        <a href="https://github.com/danielefavi/ai-db-doc-agents">
            <img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub Repo">
        </a>
        """,
        unsafe_allow_html=True
    )

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)



user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    history_to_pass = []
    # Iterate through the history, pairing HumanMessage and AIMessage
    temp_history = st.session_state.chat_history # Get the history *before* the current user query
    # Skip the initial AIMessage
    if len(temp_history) > 1:
        i = 1 # Index of the first HumanMessage
        while i + 1 < len(temp_history):
            human_msg = temp_history[i]
            ai_msg = temp_history[i+1]
            # Ensure correct types before appending
            if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                history_to_pass.append((human_msg.content, ai_msg.content))
            else:
                # Handle cases of unexpected order or if the loop logic needs adjustment
                print(f"Warning: Skipping unexpected message types/order at history index {i}")
            i += 2 # Move to the next pair

    with st.chat_message("AI"):
        response_dict = st.session_state.ai_assistant.invoke(
            user_question=user_query,
            user_chat_history=history_to_pass
        )

        message_content = f"**Response:** {response_dict['response']}\n\n" \
                        f"**SQL query executed:**\n```sql\n{response_dict['query']}\n```\n\n" \
                        f"**Result of the SQL query:**\n```\n{response_dict['result']}\n```"
        st.markdown(message_content)
        
    st.session_state.chat_history.append(AIMessage(content=response_dict['response']))