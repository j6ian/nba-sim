#!/bin/zsh
# Sets up the virtual environment on first run, then launches the app in your browser.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/streamlit run app.py
