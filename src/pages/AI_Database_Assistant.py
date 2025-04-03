import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
import streamlit as st

from sql_query_assistant import SQLQueryAssistant


# LLM_NAME = "gemma3:12b"
# LLM_NAME = "qwen2.5-coder:14b"
LLM_NAME = os.getenv("LLM_MODEL") # @TODO: it should be set from the UI


llm_model = ChatOllama(model=LLM_NAME)

DB_DETAILS = {
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASS"),
    'database': os.getenv("DB_NAME")
}

aiAssistant = SQLQueryAssistant(db_config=DB_DETAILS, llm_model=llm_model)

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
    st.markdown(
        """
        <a href="https://github.com/danielefavi/ai-db-doc-agents" target="_blank" style="text-decoration: none; color: inherit;">
            View Source Code
        </a>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---") 

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=20, max_value=80, value=50, step=5)



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
        response_dict = aiAssistant.invoke(
            user_question=user_query,
            user_chat_history=history_to_pass
        )

        message_content = f"**Response:** {response_dict['response']}\n\n" \
                        f"**SQL query executed:**\n```sql\n{response_dict['query']}\n```\n\n" \
                        f"**Result of the SQL query:**\n```\n{response_dict['result']}\n```"
        st.markdown(message_content)
        
    st.session_state.chat_history.append(AIMessage(content=response_dict['response']))