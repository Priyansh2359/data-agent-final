import os
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from agent_logic import create_agent
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

AGENT_TIMEOUT = 170

@app.post("/")
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
        
        for key, value in form_data.items():
            if key != "questions.txt" and isinstance(value, UploadFile) and value.filename:
                data_file = value
                try:
                    data_content = (await data_file.read()).decode("utf-8")
                except UnicodeDecodeError:
                    data_content = None
                break

        agent_executor = create_agent(
            file_path=data_file.filename if data_file else None, 
            file_content=data_content
        )
        
        async def run_agent_query(agent, question):
            try:
                response = await agent.ainvoke({"input": question})
                return response["output"]
            except Exception as e:
                logger.error(f"Agent execution error: {e}", exc_info=True)
                return f'{{"error": "An error occurred during agent execution.", "details": "{str(e)}"}}'

        task = asyncio.create_task(run_agent_query(agent_executor, question_content))
        result_str = await asyncio.wait_for(task, timeout=AGENT_TIMEOUT)

        logger.info(f"Agent raw output: {result_str}")

        # Try to parse the string as JSON
        try:
            final_json = json.loads(result_str)
            return JSONResponse(content=final_json)
        except json.JSONDecodeError:
            # If it's not a valid JSON string, return it as is within a structured error
            logger.error(f"Agent output was not valid JSON. Returning raw output.")
            return JSONResponse(status_code=500, content={"error": "Agent did not return valid JSON.", "output": result_str})

    except asyncio.TimeoutError:
        logger.error("Agent query timed out.")
        raise HTTPException(status_code=504, detail="The request timed out after 3 minutes.")
    except Exception as e:
        logger.exception("An unhandled error occurred in analyze_data.")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Data Analyst Agent is running."}