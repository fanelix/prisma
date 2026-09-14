@echo off
REM Klik ganda berkas ini untuk menjalankan aplikasi (Windows)
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv
    call .venv\\Scripts\\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\\Scripts\\activate.bat
)
streamlit run app.py
