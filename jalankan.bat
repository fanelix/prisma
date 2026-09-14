@echo off
REM Jalankan aplikasi Streamlit lokal (Windows): klik ganda berkas ini.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
streamlit run app\app.py
