import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from libs.chroma_loader import ChromaLoader
from libs.rag_assistant import RAGAssistant
from libs.ui_components import render_ollama_model_selector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from typing import List

load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
DOCS_FOLDER = os.getenv("DOCS_DIR")
CHUNK_SIZE = 1000 # @TODO: it should be set from the UI
CHUNK_OVERLAP = 100 # @TODO: it should be set from the UI
PROJECT_ROOT = os.getenv("PROJECT_ROOT") # @TODO: get it automatically instead from .ENV


embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

@st.cache_resource
def get_chroma_loader():
    return ChromaLoader(
        embeddings_model=embeddings_model,
        persistent_db_folder=EMBEDDING_MODEL,
        project_root=PROJECT_ROOT
    )

if 'llm_model' not in st.session_state:
    st.session_state.llm_model = ChatOllama(model=DEFAULT_LLM_MODEL)
if 'loader' not in st.session_state:
    st.session_state.loader = get_chroma_loader()
if 'agent' not in st.session_state:
    st.session_state.agent = RAGAssistant(
        embeddings_model=embeddings_model,
        llm_model=st.session_state.llm_model,
        persistent_db_folder=EMBEDDING_MODEL,
        project_root=PROJECT_ROOT
    )

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[BaseMessage] = []

# --- Streamlit UI ---
st.set_page_config(
    page_title="Chat to Docs",
    page_icon=":speech_balloon:",
    menu_items={
        'Get Help': None, # You can set a URL for a help page if you have one
        'Report a bug': None, # You can set a URL for bug reporting (e.g., GitHub issues)
        'About': f"""
        ## Chat to Database AI Assistant

        This application allows you to interact with a database using natural language queries,
        powered by AI.

        **Source Code:** [View on GitHub](https://github.com/danielefavi/ai-db-doc-agents)
        """
        # The 'About' section uses Markdown. Add any other info you want here.
    }
)
st.title("Chat to Docs")

# --- Sidebar ---
with st.sidebar:
    selected_llm = render_ollama_model_selector(
        default_model_name=DEFAULT_LLM_MODEL
    )
    st.session_state.llm_model = ChatOllama(model=selected_llm)
    st.session_state.agent = RAGAssistant(
        embeddings_model=embeddings_model,
        llm_model=st.session_state.llm_model,
        persistent_db_folder=EMBEDDING_MODEL,
        project_root=PROJECT_ROOT
    )

    st.header("Document Management")
    if st.button("⚠️ Delete Vector Store", use_container_width=True):
        with st.spinner("Deleting the vector store..."):
            try:
                st.session_state.loader.remove_vector_store()
                # Important: Clear the cached agent if the store changes!
                st.cache_resource.clear()
                st.success("Vector store deleted! Agent cache cleared.")
                # Reload the page to re-initialize the agent if needed
                # st.rerun()
            except Exception as e:
                st.error(f"Error deleting vector store: {e}")

    if st.button("Load Documents", use_container_width=True):
        # Check if docs folder exists
        if not os.path.isdir(DOCS_FOLDER):
             st.error(f"Error: Documents folder '{DOCS_FOLDER}' not found.")
        else:
            with st.spinner(f"Loading documents from '{DOCS_FOLDER}'..."):
                try:
                    st.session_state.loader.load(
                        docs_rel_folder=DOCS_FOLDER,
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    # Important: Clear the cached agent if the store changes!
                    st.cache_resource.clear()
                    st.success("Documents loaded! Agent cache cleared.")
                     # Reload the page to re-initialize the agent if needed
                    # st.rerun()
                except Exception as e:
                    st.error(f"Error loading documents: {e}")

    st.markdown("---")

    st.code(f"""
        DEFAULT_LLM_MODEL="{DEFAULT_LLM_MODEL}"
        SELECTED_LLM="{selected_llm}"
        EMBEDDING_MODEL="{EMBEDDING_MODEL}"
        DOCS_FOLDER="{DOCS_FOLDER}"
        CHUNK_SIZE={CHUNK_SIZE}
        CHUNK_OVERLAP={CHUNK_OVERLAP}
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
# --- Chat Interface ---

with st.chat_message("AI"):
    st.markdown("Hello! I'm the AI Doc assistant, how can I help you?")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Handle user input
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    # Add user message to history and display it immediately
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Prepare chat history for the agent (pass the list of BaseMessage objects)
    # The agent's invoke method expects the history *before* the current user query
    history_for_agent = st.session_state.chat_history[:-1]

    # Ensure it's a list of BaseMessage objects (simple check)
    if not all(isinstance(msg, BaseMessage) for msg in history_for_agent):
        st.error("Internal Error: Chat history format is incorrect.")
        print("Error: history_for_agent contains non-BaseMessage items:", history_for_agent)
    else:
        # Display AI response block and call the agent
        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                try:
                    print("--- USER QUERY ---")
                    print(st.session_state.llm_model.model)
                    print(user_query);
                    print("--------------------------")
                    response = st.session_state.agent.invoke(
                        user_query=user_query,
                        chat_history=history_for_agent
                    )

                    print("--- Agent Raw Response ---")
                    print(response)
                    print("--------------------------")

                    ai_answer = response.get("answer", "Sorry, I couldn't generate a response.")

                    st.markdown(ai_answer)

                    st.session_state.chat_history.append(AIMessage(content=ai_answer))

                except Exception as e:
                    st.error(f"Error getting response from AI: {e}")
                    print(f"Error during agent invocation: {e}")
                    import traceback
                    traceback.print_exc()

                    error_message = "Sorry, I encountered an error trying to respond."
                    st.markdown(error_message)
                    st.session_state.chat_history.append(AIMessage(content=error_message))