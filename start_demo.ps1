# Start the Multi-Agent Latency Demo FastAPI Server
echo "======================================================================"
echo "Starting Legal Multi-Agent Latency Demo Server"
echo "URL: http://localhost:8080"
echo "======================================================================"
echo ""

if (Test-Path ".venv\Scripts\python.exe") {
    & .venv\Scripts\python.exe latency_demo_server.py
} else {
    echo "ERROR: Virtual environment not found. Please create it or run using your system python:"
    echo "python latency_demo_server.py"
}
