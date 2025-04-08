# ui_components.py
import streamlit as st
import subprocess

# Function to get installed Ollama models (copied from your original script)
def get_ollama_models():
    """Fetches the list of locally installed Ollama models."""
    models = []
    try:
        # Added timeout for safety
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True, timeout=10)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1: # Check if there are models listed beyond the header
            # Extract model names, handling potential extra whitespace
            models = [line.split()[0] for line in lines[1:] if line.strip()] # Skip header line and empty lines
        if not models:
            # Use st.sidebar context if called from there, otherwise st
            # Using st.warning instead of st.sidebar.warning to ensure it appears even if not called within sidebar context
            st.warning("No models found by 'ollama list'. Make sure models are installed.")
    except FileNotFoundError:
        st.error("Ollama command not found. Make sure Ollama is installed and in your system's PATH.")
    except subprocess.TimeoutExpired:
         st.error("Command 'ollama list' timed out after 10 seconds.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error running 'ollama list': Check Ollama service.\n{e.stderr}")
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching models: {e}")

    return models # Return potentially empty list

def render_ollama_model_selector(
    session_state_key="selected_ollama_model",
    default_model_name=None # <-- New optional parameter
):
    """
    Renders a selectbox in the Streamlit sidebar for choosing an Ollama model.

    Args:
        session_state_key (str): The key to use for storing the selected
                                  model in st.session_state. Allows multiple
                                  independent selectors if needed.
        default_model_name (str, optional): The name of the model to select
                                             by default if available. Defaults to None,
                                             which means the first available model
                                             will be selected initially.

    Returns:
        str or None: The name of the selected Ollama model, or None if no
                     model is available or selected.
    """
    # It's generally better practice to call get_ollama_models outside the
    # `with st.sidebar:` block if the list might be needed elsewhere,
    # but for a self-contained component, calling it here is fine.
    # Error/warning messages from get_ollama_models will appear in the main area.
    available_models = get_ollama_models()

    with st.sidebar: # Ensure this component is always placed in the sidebar
        # Determine the effective default/initial value
        initial_index = 0 # Default to the first model
        if available_models:
            if default_model_name and default_model_name in available_models:
                try:
                    initial_index = available_models.index(default_model_name)
                except ValueError:
                    # Should not happen due to 'in' check, but safety first
                    pass # Keep initial_index = 0
            # Use the determined index to set the initial value in session state
            # only if the key doesn't exist yet.
            if session_state_key not in st.session_state:
                st.session_state[session_state_key] = available_models[initial_index]

            # Ensure the current selection in session state is still valid
            # If not, reset it (prefer the specified default if valid, else first)
            current_selection = st.session_state.get(session_state_key)
            if current_selection not in available_models:
                 reset_value = available_models[initial_index] # Reset to determined default
                 st.session_state[session_state_key] = reset_value
                 # Optional: Add a warning if the previous selection disappeared
                 # st.warning(f"Previously selected model '{current_selection}' not found. Resetting.")


            # Find the index of the current value in session state for the selectbox
            # This handles cases where the session state was already set (e.g. user interaction)
            try:
                 current_index = available_models.index(st.session_state[session_state_key])
            except ValueError:
                 # If the value in session state somehow became invalid *after* the check above
                 # (less likely but possible with complex interactions), default to initial_index
                 current_index = initial_index
                 st.session_state[session_state_key] = available_models[current_index]


            # Create the selectbox
            selected_model = st.selectbox(
                'Choose an Ollama model',
                options=available_models,
                index=current_index, # Use index to control the default selection
                key=session_state_key, # Link widget state to session state
                # disabled=(not available_models) # Disabling handled by checking available_models
            )
        else:
            # Handle case where no models are available
            st.info("No Ollama models available for selection.")
            # Ensure session state reflects no selection
            if session_state_key in st.session_state:
                st.session_state[session_state_key] = None
            selected_model = None # Explicitly set return value to None


        # The selectbox widget updates st.session_state[session_state_key] automatically
        # Return the currently selected value from session state
        return st.session_state.get(session_state_key)

# --- Example Usage (in your main app script) ---
# import streamlit as st
# from ui_components import render_ollama_model_selector
#
# st.title("My Ollama App")
#
# # Render the selector, trying to default to 'llama3.1:8b'
# selected_llm = render_ollama_model_selector(
#     session_state_key="my_main_llm", # Use a specific key if needed
#     default_model_name="llama3.1:8b" # Pass the desired default here
# )
#
# if selected_llm:
#     st.write(f"You selected: {selected_llm}")
# else:
#     st.write("Please install Ollama models to proceed.")
#
# # You could have another selector with a different default/key
# # selected_llm_2 = render_ollama_model_selector(
# #     session_state_key="my_secondary_llm",
# #     default_model_name="phi3:mini"
# # )
# # if selected_llm_2:
# #    st.sidebar.write(f"Secondary model: {selected_llm_2}")