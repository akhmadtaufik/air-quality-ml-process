#!/bin/bash
# Start both API and UI services

echo "=========================================="
echo "Starting Air Quality Prediction Services"
echo "=========================================="
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "^pacmann "; then
    echo "❌ Conda environment 'pacmann' not found"
    echo "   Please create it first or update the CONDA_ENV in Makefile"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $API_PID 2>/dev/null
    kill $UI_PID 2>/dev/null
    echo "✓ Services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start API in background
echo "1. Starting FastAPI server..."
conda run -n pacmann uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
API_PID=$!
echo "   ✓ API started (PID: $API_PID)"
echo "   → http://localhost:8000"
echo "   → http://localhost:8000/docs"
sleep 3

# Check if API started successfully
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "   ⚠️  API may not be ready yet, waiting..."
    sleep 2
fi

# Start Streamlit UI in background  
echo ""
echo "2. Starting Streamlit UI..."
conda run -n pacmann streamlit run src/serving/ui.py --server.headless true > ui.log 2>&1 &
UI_PID=$!
echo "   ✓ UI started (PID: $UI_PID)"
echo "   → http://localhost:8501"

echo ""
echo "=========================================="
echo "✅ Services are running!"
echo "=========================================="
echo ""
echo "Access points:"
echo "  - API:  http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - UI:   http://localhost:8501"
echo ""
echo "Logs:"
echo "  - API:  tail -f api.log"
echo "  - UI:   tail -f ui.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for both processes
wait
