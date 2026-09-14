#!/usr/bin/env bash
# Jalankan aplikasi (Linux/macOS):  bash jalankan.sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip && pip install -r requirements.txt
else
  source .venv/bin/activate
fi
streamlit run app.py
