# AI Data Analyst Agent

This project is a fully autonomous AI Data Analyst Agent. It exposes a powerful API that leverages Large Language Models (LLMs) to understand and execute complex data analysis tasks on the fly. The agent can source data from the web, prepare it, perform statistical analysis, and generate visualizations based on a simple natural language prompt.

## 🚀 Features

-   **Dynamic Task Execution:** Understands natural language requests and generates Python code to solve them.
-   **Data Sourcing:** Can scrape data from web pages.
-   **File Analysis:** Accepts and analyzes uploaded files (e.g., `.csv`) and uses them as a primary data source.
-   **Advanced Data Analysis:** Uses `pandas` and `scikit-learn` to perform calculations, correlations, and other statistical analyses.
-   **Dynamic Visualization:** Generates plots and charts using `matplotlib` and returns them as base64-encoded data URIs.
-   **API-Based:** Built with FastAPI, making it easy to integrate with other services.

## 🛠️ Tech Stack

-   **Backend:** Python, FastAPI
-   **AI / Agent Framework:** LangChain
-   **LLM Provider:** Groq (using `llama3-8b-8192`)
-   **Core Data Libraries:** Pandas, Matplotlib
-   **Deployment:** Render / Vercel

## ⚙️ Setup and Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/Priyansh2359/data-agent-final.git](https://github.com/Priyansh2359/data-agent-final.git)
cd data-agent-final
# Create a virtual environment
python -m venv venv

# Activate it (on Windows PowerShell)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
uvicorn main:app --reload
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
[https://en.wikipedia.org/wiki/List_of_highest-grossing_films](https://en.wikipedia.org/wiki/List_of_highest-grossing_films)

Answer the following questions and respond with a JSON array of strings containing the answer.
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
curl.exe -X POST "[http://127.0.0.1:8000/api/](http://127.0.0.1:8000/api/)" -F "questions.txt=@question.txt"
