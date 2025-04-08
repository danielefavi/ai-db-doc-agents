import re
import os
from typing import Dict, Optional, Any, Tuple, List # Added List, Tuple

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableAssign
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Import message types if using a more structured history format
# from langchain_core.messages import AIMessage, HumanMessage

# Define type aliases for better readability
InvokeResult = Dict[str, Any] # Contains 'response', 'query', 'result'
# Define a type for chat history (e.g., list of user/AI message tuples)
ChatHistory = List[Tuple[str, str]] # Or use List[BaseMessage] if using LangChain messages

class SQLQueryAssistant:
    """
    A class to interact with a SQL database using a large language model (LLM)
    to translate natural language questions into SQL queries and execute them,
    considering optional chat history.

    Returns the natural language response, the generated SQL query, and the raw query result.
    """

    def __init__(self, db_config: Dict[str, Any], llm_model: BaseChatModel):
        """
        Initializes the SQLQueryAssistant.

        Args:
            db_config (Dict[str, Any]): A dictionary containing database connection details.
                                        Expected keys: 'user', 'password', 'database'.
                                        Optional keys: 'host' (default '127.0.0.1'), 'port' (default 3306).
            llm_model (BaseChatModel):  The BaseChatModel instance of the LLM model to use).
        """
        if not all(key in db_config for key in ['user', 'password', 'database']):
            raise ValueError("db_config must contain 'user', 'password', and 'database'.")

        # --- Database Setup ---
        db_user = db_config['user']
        db_password = db_config['password']
        db_name = db_config['database']
        db_host = db_config.get('host', '127.0.0.1')
        db_port = db_config.get('port', 3306)

        self.mysql_uri = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        try:
            self.db = SQLDatabase.from_uri(self.mysql_uri)
            print("Database connection successful.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            print(f"Connection URI used: mysql+mysqlconnector://{db_user}:***@{db_host}:{db_port}/{db_name}")
            raise

        self.llm = llm_model

        # --- LangChain Tool and Chain Setup ---
        self._setup_sql_chain()
        self._setup_full_chain() # Setup the full chain *after* sql_chain

    def _setup_sql_chain(self):
        """Sets up the chain to generate and execute SQL.
           This chain expects an input dictionary containing 'question'
           and optionally 'chat_history'.
        """
        print("Setting up SQL generation and execution chain...")
        self.execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        # create_sql_query_chain inherently handles 'question' and 'chat_history' if present
        self.write_query_chain = create_sql_query_chain(self.llm, self.db)
        self.clean_sql_runnable = RunnableLambda(self._clean_sql_query)

        self.sql_chain = (
            self.write_query_chain # Input: {'question': ..., 'chat_history': ...}
            | self.clean_sql_runnable
            # This RunnableParallel directly outputs the desired dictionary
            | RunnableParallel(
                sql_query=RunnablePassthrough(),    # Input is the cleaned SQL string
                sql_result=self.execute_query_tool  # Input is the cleaned SQL string
            )
        )
        print("SQL generation and execution chain configured.")
        print("sql_chain input expected: Dict{'question': str, 'chat_history': Optional[ChatHistory]}")
        print("sql_chain output type: Dict{'sql_query': str, 'sql_result': Any}")


    # Helper function to format chat history for the prompt
    @staticmethod
    def _format_chat_history_string(history: Optional[ChatHistory]) -> str:
        if not history:
            return "No previous conversation history."
        formatted = []
        for user_msg, ai_msg in history:
            formatted.append(f"User: {user_msg}\nAssistant: {ai_msg}")
        return "\n".join(formatted)

    def _setup_full_chain(self):
        """
        Sets up the full chain including SQL execution and natural language response generation.
        It accepts 'question' and 'chat_history' as input.
        The chain will output a dictionary containing the original question, the schema,
        the SQL query/result dictionary, the chat history, and the final natural language response.
        """
        print("Setting up the full chain (SQL -> NL Response)...")
        # Updated prompt to include chat history context
        answer_prompt = ChatPromptTemplate.from_template(
            """Given the following user question, chat history, corresponding SQL query and SQL result, write a natural language answer that considers the conversation context.

            Chat History:
            {chat_history}

            User Question: {question}
            SQL Query: ```sql
            {query}
            ```
            SQL Result: {result}

            Natural Language Answer:"""
        )

        # This function formats the input for the answer_prompt
        def format_answer_prompt_input(input_dict: Dict) -> Dict:
            # input_dict contains: 'original_question', 'passed_chat_history', 'sql_query_result', 'schema'
            return {
                "question": input_dict['original_question'],
                "chat_history": self._format_chat_history_string(input_dict['passed_chat_history']), # Format history
                "query": input_dict['sql_query_result']['sql_query'],
                "result": input_dict['sql_query_result']['sql_result'],
            }

        # Chain to generate the natural language response string
        generate_nl_response_chain = (
            RunnableLambda(format_answer_prompt_input, name="FormatAnswerPromptInput")
            | answer_prompt
            | self.llm
            | StrOutputParser()
        )

        # The full chain:
        # 1. Starts with the input {"question": ..., "chat_history": ...}
        # 2. Uses RunnableParallel to:
        #    a. Pass the original question through.
        #    b. Pass the chat history through.
        #    c. Execute the sql_chain (which receives the full input dict) to get the SQL query/result dictionary.
        #    d. Fetch the schema.
        # 3. Uses RunnableAssign to add the 'natural_language_response' key/value
        #    by running the generate_nl_response_chain on the dictionary from step 2.
        # 4. The final output is a dictionary containing:
        #    'original_question', 'passed_chat_history', 'sql_query_result' (dict), 'schema', 'natural_language_response'

        self.full_chain = (
            RunnableParallel(
                # Extracts 'question' from the input dict {'question': ..., 'chat_history': ...}
                original_question=RunnablePassthrough() | RunnableLambda(lambda x: x['question'], name="ExtractQuestion"),
                # Extracts 'chat_history' from the input dict
                passed_chat_history=RunnablePassthrough() | RunnableLambda(lambda x: x.get('chat_history'), name="ExtractChatHistory"),
                # sql_chain receives the full input dict {'question': ..., 'chat_history': ...}
                sql_query_result=self.sql_chain,
                 # schema runnable doesn't need specific input from the dict
                schema=RunnableLambda(self.get_schema, name="GetSchema")
            ).with_config(run_name="PrepareSQLSchemaAndHistory") # Renamed step
            | RunnableAssign(
                # generate_nl_response_chain receives the output dict from the previous step
                mapper={'natural_language_response': generate_nl_response_chain}
            ).with_config(run_name="GenerateNaturalLanguageResponse")
        )
        print("Full chain configured. Input includes 'question' and optional 'chat_history'.")
        print("Full chain output includes NL response, query dict, result, history, etc.")

    def get_schema(self, _=None) -> str:
        """Fetches the database schema."""
        try:
            schema = self.db.get_table_info()
            return schema
        except Exception as e:
            print(f"Error fetching schema: {e}")
            raise

    @staticmethod
    def _clean_sql_query(text: str) -> str:
        """Extracts the MySQL query, removing potential markdown."""
        # Regex to find SQL block
        match = re.search(r'```(?:sql\s*)?(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_query = match.group(1).strip()
            if cleaned_query.endswith(';'):
                cleaned_query = cleaned_query[:-1]
            return cleaned_query
        else:
            # Fallback: Try to extract common SQL commands if no markdown block found
            lines = text.strip().split('\n')
            sql_keywords = ['SELECT', 'UPDATE', 'INSERT', 'DELETE', 'WITH', 'DESCRIBE', 'SHOW', 'ALTER', 'CREATE', 'DROP']
            sql_lines = []
            in_sql_block = False
            for line in lines:
                stripped_line_upper = line.strip().upper()
                # Check if line starts with a keyword or continues a previous SQL line
                if any(stripped_line_upper.startswith(kw) for kw in sql_keywords) or \
                   (in_sql_block and line.strip()): # Allow indented lines or continuation
                    sql_lines.append(line.strip())
                    in_sql_block = True
                else:
                    in_sql_block = False # Reset if a non-SQL line is encountered

            if sql_lines:
                cleaned_query = "\n".join(sql_lines).strip()
                if cleaned_query.endswith(';'):
                    cleaned_query = cleaned_query[:-1]
                # Handle potential prefix like "SQLQuery:"
                if cleaned_query.lower().startswith("sqlquery:"):
                   cleaned_query = cleaned_query[len("sqlquery:"):].strip()
                return cleaned_query
            else:
                # If still nothing, return the stripped text, potentially cleaned
                cleaned_query = text.strip()
                if cleaned_query.lower().startswith("sqlquery:"):
                   cleaned_query = cleaned_query[len("sqlquery:"):].strip()
                if cleaned_query.endswith(';'):
                    cleaned_query = cleaned_query[:-1]
                print(f"Warning: Could not reliably extract SQL from text: '{text}'. Returning best guess: '{cleaned_query}'")
                return cleaned_query

    def invoke(self, user_question: str, user_chat_history: Optional[ChatHistory] = None) -> InvokeResult:
        """
        Takes a user's natural language question and optional chat history,
        generates/executes SQL, generates a natural language response,
        and returns the response, query, and result.

        Args:
            user_question (str): The natural language question about the database.
            user_chat_history (Optional[ChatHistory]): A list of (user_message, ai_message) tuples
                                                      representing the conversation history. Defaults to None.

        Returns:
            InvokeResult (Dict[str, Any]): A dictionary containing:
                - 'response': The final natural language response (str).
                - 'query': The generated SQL query (str).
                - 'result': The raw result from executing the SQL query (Any).
        """
        if not user_question or not isinstance(user_question, str):
            raise ValueError("user_question must be a non-empty string.")

        # Validate chat history format if provided (basic check)
        if user_chat_history is not None and not isinstance(user_chat_history, list):
            #  print(f"Warning: user_chat_history is not a list (type: {type(user_chat_history)}). Proceeding, but ensure correct format.")
            raise ValueError("user_chat_history must be a list of tuples or None.")

        print(f"\nProcessing question: {user_question}")
        if user_chat_history:
            print(f"With chat history: {self._format_chat_history_string(user_chat_history)}") # Use formatter for clean log

        try:
            # Prepare the input dictionary for the full_chain
            chain_input = {
                "question": user_question,
                "chat_history": user_chat_history # Pass history (can be None)
            }

            # Invoke the full chain with the combined input
            chain_output: Dict = self.full_chain.invoke(chain_input)

            # Extract the components from the potentially nested structure
            final_response = chain_output.get('natural_language_response', 'Error: Could not generate response')
            sql_details: Optional[Dict] = chain_output.get('sql_query_result') # This comes from the parallel step

            if sql_details and isinstance(sql_details, dict):
                generated_query = sql_details.get('sql_query', "Error: 'sql_query' key missing")
                query_result = sql_details.get('sql_result', "Error: 'sql_result' key missing")
            else:
                generated_query = "Error: Failed to generate/execute query or result format incorrect"
                query_result = "Error: Query execution details missing or malformed"
                print(f"Warning: Expected a dictionary for 'sql_query_result', but got: {type(sql_details)}")

            print(f"\nGenerated SQL Query:\n---\n{generated_query}\n---")
            print(f"\nSQL Query Raw Result:\n---\n{query_result}\n---")
            print(f"\nFinal Natural Language Response:\n---\n{final_response}\n---")

            # Return the desired dictionary structure
            return {
                "response": final_response,
                "query": generated_query,
                "result": query_result,
            }
        except Exception as e:
            print(f"An error occurred during the invocation chain: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for detailed debugging
            raise
