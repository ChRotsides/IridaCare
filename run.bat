@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:8080
".venv\Scripts\python.exe" app.py
