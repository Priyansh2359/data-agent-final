import os
import pandas as pd
import base64
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
# Import ChatOpenAI because Groq's API is OpenAI-compatible
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

    # This is the detailed instruction prompt for the agent
    template = f"""
    You are a powerful data analyst agent. Your goal is to answer the user's questions accurately.
    You have access to a Python code execution tool. Use it to perform any necessary data sourcing,
    manipulation, analysis, and visualization.

    CONTEXT:
    {file_context}
    
    TOOLS:
    ------
    You have access to the following tools:
    {{tools}}
    
    To use a tool, please use the following format:
    ```
    Thought: Do I need to use a tool? Yes
    Action: The action to take.
    Action Input: The input to the action.
    Observation: The result of the action.
    ```
    
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```
    
    INSTRUCTIONS:
    1.  **Analyze the user's question carefully.**
    2.  **Write and execute Python code** to find the answer. You can use libraries like pandas, numpy, scikit-learn, matplotlib, duckdb, requests, BeautifulSoup, etc.
    3.  **Web Scraping**: If the question involves a URL, use the `requests` and `BeautifulSoup` libraries to scrape data.
    4.  **Plotting**: If a plot is requested, generate it using `matplotlib`.
        -   **IMPORTANT**: Do NOT use `plt.show()`. Instead, save the plot to a BytesIO object and encode it.
        -   Encode the plot image as a base64 data URI string in the format: `data:image/png;base64,...`.
        -   Ensure the final data URI is less than 100,000 bytes. Use a lower DPI if needed (e.g., `dpi=75`).
        -   Make sure the plot matches the request exactly (e.g., scatterplot, dotted red regression line, labels).
    5.  **Final Answer Format**:
        -   Your final answer MUST be only the JSON structure requested by the user.
        -   Do not include any other text, explanations, or markdown formatting like ```json. Just the raw JSON.
        -   Example (JSON Array): ["answer1", 2, 3.14, "data:image/png;base64,..."]
    
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