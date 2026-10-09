from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage

from .agent import research_agent
from .ingestion import build_or_update_vector_db

app = FastAPI(
    title="LocalLens - 100% Offline Research Agent API",
    description="Privacy-first, Zero-API-Cost Research Agent powered by Ollama, Qdrant & LangGraph.",
    version="1.0.0"
)

class ResearchRequest(BaseModel):
    query: str = Field(..., example="What are the safety altitude guidelines for Project Apollo?")

class ToolStep(BaseModel):
    tool: str
    input: dict

class ResearchResponse(BaseModel):
    query: str
    answer: str
    tool_steps: List[ToolStep] = []


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "online",
        "llm_engine": "Ollama (llama3)",
        "vector_store": "Qdrant (Embedded Local Disk)",
        "mode": "100% Free / Air-Gapped Capable"
    }


@app.post("/ingest", tags=["Knowledge Base"])
def trigger_ingestion():
    """Re-indexes all files in the data/docs directory into the local vector DB."""
    try:
        store = build_or_update_vector_db()
        if not store:
            return {"status": "warning", "message": "No documents found in data/docs to ingest."}
        return {"status": "success", "message": "Documents successfully parsed and indexed in local Qdrant."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research", response_model=ResearchResponse, tags=["Agent"])
def research_endpoint(payload: ResearchRequest):
    """
    Submits a query to the LangGraph ReAct agent.
    The agent autonomously decides whether to query private documents or search the web.
    """
    try:
        initial_state = {"messages": [HumanMessage(content=payload.query)]}
        tool_steps = []
        final_answer = ""

        for event in research_agent.stream(initial_state, stream_mode="values"):
            last_msg = event["messages"][-1]

            # Collect tool execution calls
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    tool_steps.append(ToolStep(tool=tc["name"], input=tc["args"]))

            # Extract final text answer
            if last_msg.type == "ai" and not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                final_answer = last_msg.content

        return ResearchResponse(
            query=payload.query,
            answer=final_answer,
            tool_steps=tool_steps
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(e)}"
        )

