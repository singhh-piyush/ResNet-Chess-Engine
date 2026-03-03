#!/bin/bash

# Kill any existing processes (optional but helps cleanup)
pkill -f "python main.py"
pkill -f "vite"

echo "Starting Chess Bot Project..."

# Start Backend
echo "Step 1: Starting Backend (Port 8000)..."
if [ -d "venv" ]; then
    source venv/bin/activate
    python main.py &
    BACKEND_PID=$!
    echo "Backend started with PID $BACKEND_PID"
else
    echo "Error: Virtual environment 'venv' not found!"
    exit 1
fi

# Wait a bit for backend to init
sleep 2

# Start Frontend
echo "Step 2: Starting Frontend (Port 5173)..."
cd chess-frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend started with PID $FRONTEND_PID"

echo "Project is running!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop both."

# Handle shutdown
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

wait
