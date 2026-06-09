import asyncio
import contextlib
import io
import os
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Ensure Workspace Root is in Python Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load Environment Variables
from dotenv import load_dotenv
load_dotenv()

# Import the graph creators
from stages.stage_4_milti_agent.main_instrumented import create_graph as create_unoptimized_graph
from stages.stage_4_milti_agent.main_optimized import create_graph as create_optimized_graph

# Compile graphs once on startup
try:
    unopt_graph = create_unoptimized_graph()
    print("Unoptimized graph compiled successfully.")
except Exception as e:
    print(f"Error compiling unoptimized graph: {e}")
    unopt_graph = None

try:
    opt_graph = create_optimized_graph()
    print("Optimized graph compiled successfully.")
except Exception as e:
    print(f"Error compiling optimized graph: {e}")
    opt_graph = None

# Initialize FastAPI App
app = FastAPI(title="Legal Multi-Agent Latency Analyzer")

# Configure CORS for local development and file access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    question: str
    mode: str  # 'unoptimized' or 'optimized'

@app.get("/", response_class=HTMLResponse)
def read_index():
    """Serve the demo dashboard HTML file."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_demo.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>agent_demo.html not found! Please run writing task first.</h1>")

@app.get("/api/health")
def health_check():
    """Simple status check for frontend dashboard."""
    return {"status": "ok"}

@app.post("/api/run")
async def run_agents(req: RunRequest):
    """Run the multi-agent system and capture execution times and logs."""
    if req.mode == "unoptimized":
        graph = unopt_graph
        if not graph:
            raise HTTPException(status_code=500, detail="Unoptimized graph could not be compiled.")
    elif req.mode == "optimized":
        graph = opt_graph
        if not graph:
            raise HTTPException(status_code=500, detail="Optimized graph could not be compiled.")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'unoptimized' or 'optimized'.")

    # Capture stdout prints during LangGraph run
    stdout_buffer = io.StringIO()
    start_time = time.time()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            result = await graph.ainvoke({
                "question": req.question,
                "law_analysis": "",
                "needs_tax": False,
                "needs_compliance": False,
                "tax_result": "",
                "compliance_result": "",
                "final_answer": "",
                "node_timings": {},
            })
        total_latency = time.time() - start_time
        logs = stdout_buffer.getvalue()
        
        return {
            "status": "success",
            "final_answer": result.get("final_answer", ""),
            "node_timings": result.get("node_timings", {}),
            "total_latency": total_latency,
            "logs": logs,
            "needs_tax": result.get("needs_tax", False),
            "needs_compliance": result.get("needs_compliance", False),
        }
    except Exception as e:
        import traceback
        error_logs = stdout_buffer.getvalue() + f"\n\nERROR: {str(e)}\n" + traceback.format_exc()
        return {
            "status": "error",
            "error": str(e),
            "logs": error_logs,
            "total_latency": time.time() - start_time,
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"Starting server on port {port}...")
    uvicorn.run("latency_demo_server:app", host="127.0.0.1", port=port, reload=True)
