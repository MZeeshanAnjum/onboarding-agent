from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn
import base64
from io import BytesIO
from agents_helper.ag2 import AgentState as State
from agents_helper.ob_agent import OnboardingAgent, AgentState

app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = OnboardingAgent()

@app.post("/invoke")
def invoke_agent(request: AgentState):
    payload = request.dict()
    print(f"Received payload: {payload}")
    try:
        # Invoke the compiled state graph. The compiled graph accepts plain dict input.
        result = agent.graph.invoke(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"result": result}

@app.post("/get_rag_agent")
def get_rag_agent():
    try:
        rag_agent = agent.rag_based_agent({})
        return {"summary": rag_agent.get("RAG_summary", ""),"unanswered_questions": rag_agent.get("unanswered_questions", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run with: python ob_app.py
    uvicorn.run("ob_app:app", host="0.0.0.0", port=8000, reload=True)
