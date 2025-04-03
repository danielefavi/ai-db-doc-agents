import os
import re
from typing import List, Dict, Any

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel 
from langchain_chroma import Chroma



class RAGAssistant:
    """
    A Retrieval-Augmented Generation (RAG) chatbot class that uses LangChain
    with Ollama models and ChromaDB for retrieval.
    """

    def __init__(self, embeddings_model: Embeddings, llm_model: BaseChatModel, persistent_db_folder: str, project_root: str):
        """
        Initializes the RAGAgent.
        """
        self.embeddings = embeddings_model
        self.llm = llm_model
        self.persistent_db_folder = persistent_db_folder

        self.project_root = os.path.abspath(project_root) # Ensure it's an absolute path
        if not os.path.isdir(self.project_root):
            raise NotADirectoryError(f"Provided project root '{self.project_root}' is not a valid directory.")

    @staticmethod
    def _sanitize_for_foldername(name: str) -> str:
        """Sanitizes a string to be suitable for use as a folder name."""
        sanitized = name.lower()
        sanitized = re.sub(r'[^\w-]', '_', sanitized)   # Replace non-alphanumeric/- characters with _
        sanitized = re.sub(r'_+', '_', sanitized)       # Replace multiple underscores with single
        sanitized = sanitized.strip('_')                # Remove leading/trailing underscores
        return sanitized

    def _setup_vector_store(self):
        """Sets up the Chroma vector store."""
        # Construct the path based on the project root and the provided folder name
        # The 'db' folder is expected to be directly inside the project root.
        # self.persistent_db_folder should be the *actual* name of the Chroma DB folder.
        folder_name = "chroma_db_" + self._sanitize_for_foldername(self.persistent_db_folder)
        persistent_directory = os.path.join(self.project_root, "db", folder_name)

        print(f"Attempting to load Chroma DB from: {persistent_directory}")

        # Ensure the target directory exists before trying to load
        if not os.path.isdir(persistent_directory):
             raise NotADirectoryError(
                 f"Chroma persistent directory not found at '{persistent_directory}'. "
                 f"Ensure the database was created correctly in this location."
            )

        # Load the existing vector store with the embedding function
        try:
            self.db = Chroma(persist_directory=persistent_directory, embedding_function=self.embeddings)
            print(f"Successfully loaded Chroma DB from: {persistent_directory}")
        except Exception as e:
            print(f"Error loading Chroma DB from {persistent_directory}: {e}")
            raise e

    def _setup_retriever(self, search_type: str = "similarity", k: int = 3):
        """Creates the retriever from the vector store."""
        print(f"Setting up retriever (search_type={search_type}, k={k})")
        self.retriever = self.db.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

    def _setup_chains(self):
        """Creates the LangChain chains (history-aware retriever, QA chain, RAG chain)."""
        print("Setting up LangChain chains...")
        # --- Contextualize question ---
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # --- Answer question ---
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        doc_chain_for_model = create_stuff_documents_chain(self.llm, qa_prompt)

        # --- Create final RAG chain ---
        self.rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain_for_model)
        print("Chains setup complete.")

    def invoke(self, user_query: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """
        Processes a user query using the RAG chain, considering chat history.

        Args:
            user_query (str): The latest question or input from the user.
            chat_history (List[BaseMessage]): A list of LangChain message objects
                                              representing the conversation history.

        Returns:
            Dict[str, Any]: The result dictionary from the RAG chain, typically
                            containing keys like 'input', 'chat_history',
                            'context', and 'answer'.
        """
        if not isinstance(chat_history, list):
             raise TypeError("chat_history must be a list of BaseMessage objects.")
        # Ensure all elements in chat_history are BaseMessage instances
        if not all(isinstance(msg, BaseMessage) for msg in chat_history):
             raise TypeError("All elements in chat_history must be instances of BaseMessage (e.g., HumanMessage, SystemMessage).")

        self._setup_vector_store()
        self._setup_retriever()
        self._setup_chains()
        
        return self.rag_chain.invoke({"input": user_query, "chat_history": chat_history})
