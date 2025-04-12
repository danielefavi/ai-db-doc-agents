# Langchain AI Agents: Interact with Databases & Documents using Natural Language

This project provides two powerful **AI agents** built with Python and **Langchain**, allowing you to interact with your data sources (**MySQL database** and **documents**) using natural language. 

Keep your data private by using local LLMs or leverage powerful cloud models when needed.

---

## Features

1.  **AI Database Assistant:**
    * Query your databases using everyday language.
    * Ask questions like "Show me users older than 18" and get direct results.
    * Connects to your database and translates your questions into SQL queries (or relevant DB queries).

2.  **AI Document Assistant:**
    * Upload your documents (.md, .txt).
    * Ask questions based on the content of your documents.
    * Get answers synthesized from the information within your files.

3.  **Privacy-Focused:**
    * Designed to run with **local LLMs** by default, ensuring your database contents and documents remain on your machine.
    * No data is sent to external providers unless you explicitly configure it.

4.  **Flexible Model Choice:**
    * Optionally configure the agents to use external models like OpenAI (GPT-*) or Anthropic (Claude) for potentially more complex tasks.

---

## Screenshots

| Chat To Database | Chat To Docs |
| --- | --- |
| <img src="https://raw.githubusercontent.com/danielefavi/ai-db-doc-agents/refs/heads/main/.github-uploads/ai-agent-chat-to-database.png" /> | <img src="https://raw.githubusercontent.com/danielefavi/ai-db-doc-agents/refs/heads/main/.github-uploads/ai-agent-chat-to-docs.png" /> |

---

## Development Status

⚠️ **Please Note:** This project is currently **under development**. Features may change, and bugs might be present.

---

## Technology Stack

* **Backend:** Python
* **AI Framework:** Langchain
* **Frontend:** Streamlit
* **LLMs:** Supports local models (e.g., Llama) and external APIs (OpenAI, Anthropic, etc.)

---

## Getting Started

### Prerequisites

* Python 3.x
* Access to a terminal or command prompt.
* Ollama
    * At least LLM: in the `.env.example` you will find `llama3.1:8b`
    * One Embedding model: in the `.env.example` you will find `snowflake-arctic-embed2:568m`

### Installation & Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/danielefavi/ai-db-doc-agents.git
    cd ai-db-doc-agents
    ```

2.  **Set up Environment Variables:**
    * Rename the example environment file `.env.example` to `.env`.
    * Edit the `.env` file and add your credentials for:
        * Database connection details (e.g., `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`)
        * LangSmith tracing details (Optional)
    * Rename the name of the LLM you want to use (**it must be installed on Ollama**).
    * Rename the name of the embedding model you want to use (**it must be installed on Ollama**).

3.  **Install Dependencies:**

    **On Linux/Mac:**
    ```sh
    python3 -m venv venv
    chmod +x ./venv/bin/activate
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    Ensure to you have read/write/execute permissions to the `db` folder (where all vector stores are going to be placed):

    ```sh
    chmod -R u+rwx db
    ```

    **On Windows:**
    ```sh
    python -m venv venv
    .\venv\Scripts\activate.bat
    pip install -r requirements.txt
    ```

4.  **Run the Application:**

    Make sure Ollama is running and the LLM and embedding model you set on the `.env` are installed on your local.

    ```sh
    streamlit run src/Doc_Assistant.py
    ```

    This will start the Streamlit application, and you can access it in your web browser (usually at `http://localhost:8501`).

---

## Usage

### AI Doc Assistant

1. Copy your documents (**only `.TXT` or `.MD` files**) into the `docs` folder.
2. Press the button *Load Documents*:
    * In this phase the embedding model will create embeddings from the documents and they will be stored in the vector store.
    * This phase is going to take a while depending on the embedding model you choose and the amount of the documents.
3. Once it finishes you can ask question to the AI agent.

> **IMPORTANT NOTE** if you change the embedding model you must load again the documents!

### AI Database Assistant

1. Make sure that the database details in the `.env` are correct.
2. Make sure the database is up and running.
3. Query the AI database agent.

