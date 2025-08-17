import os
import pandas as pd
import base64
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description

# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

def create_agent(file_path: str = None, file_content: str = None):
    """
    Creates a data analyst agent that can execute Python code to answer questions.
    """
    # Initialize the LLM using Groq's API
    llm = ChatOpenAI(
        model_name="llama3-8b-8192", # A powerful model available on Groq
        temperature=0,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Prepare context about the file, if one was provided
    file_context = "No file has been provided."
    df = None
    if file_path:
        file_name = os.path.basename(file_path)
        if file_path.endswith('.csv') and file_content:
            try:
                df = pd.read_csv(StringIO(file_content))
                file_context = f"A pandas DataFrame named `df` has been loaded from the file '{file_name}'. Use this DataFrame to answer the questions."
            except Exception as e:
                file_context = f"Could not read the CSV file '{file_name}'. Error: {e}"
        else:
            file_context = f"The content of the file '{file_name}' has been provided. Analyze it to answer the questions."
    
    # Define the tools the agent can use (in this case, a Python code interpreter)
    tools = [PythonREPLTool()]
    if df is not None:
        # If we have a DataFrame, make it available within the Python tool's environment
        tools[0].locals = {"df": df}

    # This is the new, improved instruction prompt for the agent
    template = f"""
    You are a powerful data analyst agent. Your goal is to answer the user's questions accurately by writing and executing Python code.

    CONTEXT:
    {file_context}

    TOOLS:
    ------
    You have access to the following tools:
    {{tools}}

    To use a tool, you must use the following format:
    ```
    Thought: Do I need to use a tool? Yes
    Action: The action to take. Always use 'python_repl_ast'.
    Action Input: The Python code to execute.
    Observation: The result of the action.
    ```

    When you have the final answer, you MUST use the format:
    ```
    Thought: Do I need to use a tool? No
    Final Answer: [YOUR FINAL JSON RESPONSE]
    ```

    ## INSTRUCTIONS & RULES:

    1.  **THINK STEP-BY-STEP:** For each question asked, break it down. First, write the code to solve it. Second, look at the result (the Observation). Third, verify the result is correct before moving on.

    2.  **HANDLE COMPLEX TASKS EFFICIENTLY:** For specific tasks, use the recommended libraries to avoid timeouts.
        -   **Network Analysis:** Use the `networkx` library.
        -   **Web Scraping:** Use `requests` and `BeautifulSoup`.
        -   **Data Analysis:** Use `pandas`.

    3.  **GENERATE PLOTS CORRECTLY:** When asked to create a plot or chart:
        -   Use the `matplotlib` library.
        -   **NEVER use `plt.show()`**.
        -   Always save the plot to a BytesIO object and encode it as a base64 data URI string.
        -   Pay close attention to details like chart type (line, bar, histogram) and colors (red, orange, etc.).

    4.  **FINAL ANSWER FORMAT IS CRITICAL:**
        -   The final answer MUST be ONLY a single, valid JSON object or JSON array as requested.
        -   Do NOT add any extra text, notes, explanations, or markdown formatting like ```json before or after the JSON.
        -   **Correct Example:** `Final Answer: {{"key1": "value1", "key2": 123}}`
        -   **Incorrect Example:** `Final Answer: Here is the JSON you requested: {{"key1": "value1"}}`
        -   **Incorrect Example:** `Final Answer: {{"key1": "value1"}} Note: The data was clean.`

    5.  **SELF-CORRECTION:** Before giving the Final Answer, perform a final review of all your steps to ensure your calculations are correct and the output format is perfect.

    Begin!

    Previous conversation history:
    {{agent_scratchpad}}

    Question: {{input}}
    Thought:
    """
    
    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(tools=render_text_description(tools))
    
    # Chain the components together to create the agent
    agent_chain = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm.bind(stop=["\nObservation"])
        | ReActSingleInputOutputParser()
    )
    
    # Create the agent executor which runs the agent
    agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor