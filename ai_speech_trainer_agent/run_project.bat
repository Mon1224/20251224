@echo off
setlocal

REM ============================================================
REM  AI Speech Trainer - one-click launcher (Windows)
REM  Starts the backend (FastAPI + uvicorn) and the frontend
REM  (Streamlit) in two separate console windows.
REM  Close both windows (or press Ctrl+C) to stop the app.
REM ============================================================

set "ROOT=%~dp0"

REM --- Check prerequisites -------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    pause
    exit /b 1
)

python -c "import uvicorn, streamlit" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uvicorn or streamlit is not installed.
    echo         Run: pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "%ROOT%backend\main.py" (
    echo [ERROR] backend\main.py not found. Run this script from the project root.
    pause
    exit /b 1
)

if not exist "%ROOT%frontend\Home.py" (
    echo [ERROR] frontend\Home.py not found.
    pause
    exit /b 1
)

if not exist "%ROOT%.env" (
    echo [WARNING] .env not found. Copy .env.example to .env and set QWEN_API_KEY first.
)

echo ============================================================
echo  Starting AI Speech Trainer ...
echo    Backend  : http://127.0.0.1:8000
echo    Frontend : http://localhost:8501
echo  The browser should open automatically.
echo  Close the two new windows to stop the servers.
echo ============================================================

REM --- Launch backend and frontend in separate windows ---------
start "AI Speech Trainer - Backend" /D "%ROOT%backend" cmd /k "python -m uvicorn main:app --reload"
start "AI Speech Trainer - Frontend" /D "%ROOT%frontend" cmd /k "python -m streamlit run Home.py"

endlocal
