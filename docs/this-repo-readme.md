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

## Screenshot

| Chat To Database | Chat To Docs |
| --- | --- |
| <img src="https://raw.githubusercontent.com/danielefavi/ai-db-doc-agents/refs/heads/main/.github-uploads/ai-agent-chat-to-database.png" width="90%" /> | <img src="https://raw.githubusercontent.com/danielefavi/ai-db-doc-agents/refs/heads/main/.github-uploads/ai-agent-chat-to-docs.png" width="90%" /> |

---

## Development Status

⚠️ **Please Note:** This project is currently **under development**. Features may change, and bugs might be present.

* **Important:** As the project is evolving, the names of the specific LLM and Embedding models used by the agents are currently **hardcoded** within the Streamlit pages. Future versions will have model selection more dynamic, through the UI.

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
    * At least LLM
    * One Embedding model

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

3.  **Install Dependencies:**

    **On Linux/Mac:**
    ```sh
    python3 -m venv venv
    chmod +x ./venv/bin/activate
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    **On Windows:**
    ```sh
    python -m venv venv
    .\venv\Scripts\activate.bat
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    ```sh
    streamlit run src/Doc_Assistant.py
    ```

    This will start the Streamlit application, and you can access it in your web browser (usually at `http://localhost:8501`).

---

## Usage

1.  Open the application in your browser.
2.  Use the sidebar to select either the "AI Doc Assistant" or the "AI Database Assistant".
3.  Follow the on-screen instructions to:
    * Upload documents for the Doc Assistant.
    * Interact with the connected database via the chat interface for the Database Assistant.
4.  Configure model parameters (like temperature, model choice) using the sidebar controls if needed.

