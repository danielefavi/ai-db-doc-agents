import os
import re
import shutil
from typing import List, Optional

# ... (other imports remain the same)
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class ChromaLoader:
    """
    A class to handle loading documents, splitting them, creating embeddings,
    and storing them into a persistent Chroma vector database using Ollama.
    Requires the project root path to correctly locate document and database folders.
    """

    def __init__(self, embeddings_model: Embeddings, persistent_db_folder: str, project_root: str):
        """
            Initializes the ChromaLoader.

            Args:
                embeddings_model: The embeddings model to use.
                persistent_db_folder: A name suffix for the persistent db folder (e.g., model name).
                project_root: The absolute path to the project's root directory.
        """
        self.project_root = os.path.abspath(project_root) # Ensure it's an absolute path
        print(f"ChromaLoader initialized. Project root: {self.project_root}")
        if not os.path.isdir(self.project_root):
             raise NotADirectoryError(f"Provided project root '{self.project_root}' is not a valid directory.")

        self.embeddings = embeddings_model
        self.persistent_db_folder = persistent_db_folder

    def _get_abs_doc_dir(self, docs_rel_folder: str) -> str:
        """
        Calculates the absolute path for the document directory, relative to the project root.

        Args:
            docs_rel_folder: The relative path to the documents folder from the project root.

        Returns:
            The absolute path to the documents folder.
        """
        # Join the project root with the relative docs folder path
        return os.path.join(self.project_root, docs_rel_folder)

    def _sanitize_for_foldername(self, name: str) -> str:
        """Sanitizes a string to be suitable for use as a folder name."""
        sanitized = name.lower()
        sanitized = re.sub(r'[^\w-]', '_', sanitized)   # Replace non-alphanumeric/- characters with _
        sanitized = re.sub(r'_+', '_', sanitized)       # Replace multiple underscores with single
        sanitized = sanitized.strip('_')                # Remove leading/trailing underscores
        return sanitized

    def _get_persistent_db_dir(self) -> str:
        """
        Calculates the absolute path for the persistent Chroma database directory,
        named based on the persistent_db_folder suffix, located under the project root's 'db' folder.

        Returns:
            The absolute path to the persistent database directory.
        """
        db_base_dir = os.path.join(self.project_root, "db")
        folder_name = "chroma_db_" + self._sanitize_for_foldername(self.persistent_db_folder)
        return os.path.join(db_base_dir, folder_name)

    def _load_docs(self, abs_docs_dir: str) -> List[Document]:
        """Loads documents..."""
        if not os.path.exists(abs_docs_dir):
            raise FileNotFoundError(f"The document directory {abs_docs_dir} does not exist.")
        if not os.path.isdir(abs_docs_dir):
             raise NotADirectoryError(f"The path {abs_docs_dir} is not a directory.")

        allowed_extensions = (".txt", ".md")
        doc_files = [f for f in os.listdir(abs_docs_dir)
                     if os.path.isfile(os.path.join(abs_docs_dir, f)) and f.endswith(allowed_extensions)]

        print(f"Found {len(doc_files)} files with allowed extensions {allowed_extensions} in {abs_docs_dir}")

        documents = []
        for doc_file in doc_files:
            file_path = os.path.join(abs_docs_dir, doc_file)
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata = {"source": doc_file}
                documents.extend(loaded_docs)
                print(f"  - Loaded {doc_file}")
            except Exception as e:
                print(f"  - Failed to load {doc_file}: {e}")
        print(f"Successfully loaded {len(documents)} documents.")
        return documents

    def _split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """Splits the loaded documents..."""
        print(f"\nSplitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs_split = text_splitter.split_documents(documents)
        print(f"--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs_split)}")
        return docs_split

    def _store_to_vectorstore(self, docs_split: List[Document], persistent_directory: str) -> Chroma:
        """Creates and persists a Chroma vector store..."""
        print("\n--- Creating and persisting vector store ---")
        print(f"Storing to: {persistent_directory}")
        # os.makedirs(os.path.dirname(persistent_directory), exist_ok=True)
        os.makedirs(persistent_directory, exist_ok=True)
        db = Chroma.from_documents(docs_split, self.embeddings, persist_directory=persistent_directory)
        print("--- Finished creating and persisting vector store ---")
        return db

    def load(self, docs_rel_folder: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> Optional[Chroma]:
        """Orchestrates the process..."""
        abs_docs_dir = self._get_abs_doc_dir(docs_rel_folder)
        persistent_directory = self._get_persistent_db_dir()
        print(f"\nConfiguration:")
        print(f"  Document directory (absolute): {abs_docs_dir}")
        print(f"  Vector DB directory: {persistent_directory}")
        print(f"  Chunk Size: {chunk_size}")
        print(f"  Chunk Overlap: {chunk_overlap}")

        if not os.path.exists(persistent_directory):
            print("\nPersistent directory does not exist. Initializing vector store...")
            try:
                documents = self._load_docs(abs_docs_dir)
                if not documents:
                    print("No documents found or loaded. Aborting vector store creation.")
                    return None
                docs_split = self._split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                db = self._store_to_vectorstore(docs_split, persistent_directory)
                print("\nVector store initialization complete.")
                return db
            except FileNotFoundError as e:
                print(f"\nError: {e}")
                print("Vector store initialization aborted: File Not Found.")
                raise
            except NotADirectoryError as e:
                print(f"\nError: {e}")
                print("Vector store initialization aborted: Path is not a directory.")
                raise
            except Exception as e:
                print(f"\nAn unexpected error occurred during initialization: {e}")
                print("Vector store initialization aborted due to an exception.")
                import traceback
                traceback.print_exc() # Uncomment for detailed debugging
                raise
        else:
            print(f"\nVector store already exists at {persistent_directory}.")
            print("Loading existing vector store...")
            try:
                db = Chroma(persist_directory=persistent_directory, embedding_function=self.embeddings)
                print("Successfully loaded existing vector store.")
                return db
            except Exception as e:
                 print(f"\nError loading existing vector store: {e}")
                 return None

    def remove_vector_store(self) -> bool:
        """Removes the persistent vector store directory..."""
        persistent_directory = self._get_persistent_db_dir()
        print(f"\nAttempting to remove vector store directory: '{persistent_directory}'")
        if os.path.exists(persistent_directory) and os.path.isdir(persistent_directory):
            try:
                shutil.rmtree(persistent_directory)
                print(f"Successfully removed directory: {persistent_directory}")
                parent_dir = os.path.dirname(persistent_directory)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    try:
                        os.rmdir(parent_dir)
                        print(f"Successfully removed empty parent directory: {parent_dir}")
                    except OSError as e:
                        print(f"Could not remove empty parent directory {parent_dir}: {e}")
                return True
            except OSError as e:
                print(f"Error removing directory {persistent_directory}: {e}")
                return False
        else:
            print(f"Directory not found or is not a valid directory: {persistent_directory}")
            print("No action taken.")
            return False
