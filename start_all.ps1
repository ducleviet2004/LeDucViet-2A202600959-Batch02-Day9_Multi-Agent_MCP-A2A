# Start all Legal Multi-Agent System services in Windows PowerShell

$processes = @()

try {
    echo "Starting Registry service on port 10000..."
    $registry = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m registry" -NoNewWindow -PassThru
    $processes += $registry
    Start-Sleep -Seconds 2

    echo "Starting Tax Agent on port 10102..."
    $tax = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m tax_agent" -NoNewWindow -PassThru
    $processes += $tax

    echo "Starting Compliance Agent on port 10103..."
    $compliance = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m compliance_agent" -NoNewWindow -PassThru
    $processes += $compliance

    echo "Starting RAG Agent on port 10104..."
    $rag = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m rag_agent" -NoNewWindow -PassThru
    $processes += $rag
    Start-Sleep -Seconds 3

    echo "Starting Law Agent on port 10101..."
    $law = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m law_agent" -NoNewWindow -PassThru
    $processes += $law
    Start-Sleep -Seconds 3

    echo "Starting Customer Agent on port 10100..."
    $customer = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m customer_agent" -NoNewWindow -PassThru
    $processes += $customer

    echo ""
    echo "All services started:"
    echo "  Registry:         http://localhost:10000"
    echo "  Customer Agent:   http://localhost:10100"
    echo "  Law Agent:        http://localhost:10101"
    echo "  Tax Agent:        http://localhost:10102"
    echo "  Compliance Agent: http://localhost:10103"
    echo "  RAG Agent:        http://localhost:10104"
    echo ""
    echo "Press Ctrl+C to stop all services."

    # Keep script running to monitor/wait
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    echo "Stopping all services..."
    foreach ($p in $processes) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
