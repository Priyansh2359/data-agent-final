import os
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from agent_logic import create_agent
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Timeout for the agent query in seconds
AGENT_TIMEOUT = 170 # 2 minutes and 50 seconds, just under the 3 min limit

@app.post("/api/")
async def analyze_data(request: Request):
    try:
        form_data = await request.form()
        questions_file = form_data.get("questions.txt")
        if not questions_file:
            raise HTTPException(status_code=400, detail="questions.txt file is mandatory.")

        question_content = (await questions_file.read()).decode("utf-8")
        logger.info(f"Received question: {question_content}")

        data_file = None
        data_content = None
        
        # Find the first potential data file (e.g., .csv)
        for key, value in form_data.items():
            if key != "questions.txt" and isinstance(value, UploadFile) and value.filename:
                data_file = value
                try:
                    # Try to decode as text first
                    data_content = (await data_file.read()).decode("utf-8")
                    logger.info(f"Received data file: {data_file.filename}")
                except UnicodeDecodeError:
                    logger.warning(f"Could not decode file {data_file.filename} as text.")
                    await data_file.seek(0)
                    data_content = None 
                break

        agent_executor = create_agent(
            file_path=data_file.filename if data_file else None, 
            file_content=data_content
        )
        
        async def run_agent_query(agent, question):
            try:
                # Use ainvoke for async execution
                response = await agent.ainvoke({"input": question})
                return response["output"]
            except Exception as e:
                logger.error(f"Agent execution error: {e}", exc_info=True)
                return f"An error occurred during agent execution: {str(e)}"

        try:
            task = asyncio.create_task(run_agent_query(agent_executor, question_content))
            result_str = await asyncio.wait_for(task, timeout=AGENT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("Agent query timed out.")
            raise HTTPException(status_code=504, detail="The request timed out after 3 minutes.")

        logger.info(f"Agent raw output: {result_str}")

        try:
            # Clean the output in case the LLM adds extra text or markdown
            if "Final Answer:" in result_str:
                result_str = result_str.split("Final Answer:")[-1].strip()
            if result_str.startswith("```json"):
                result_str = result_str.strip("```json\n").strip("`\n")
            
            final_json = json.loads(result_str)
            return JSONResponse(content=final_json)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse agent output as JSON: {result_str}")
            return JSONResponse(status_code=500, content={"error": "Agent did not return valid JSON.", "output": result_str})

    except Exception as e:
        logger.exception("An error occurred during processing.")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Data Analyst Agent is running."}