@echo off
echo 🚀 Starting Vipo Knowledge Base Dashboard...
echo.
echo 📁 Make sure you're in the dashboard folder
echo 🔧 Installing dependencies...
pip install -r requirements.txt
echo.
echo 🌐 Starting dashboard server...
echo 📊 Dashboard will be available at: http://localhost:8001
echo 💬 Your RAG chat is at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
python main.py
pause
