#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp -n .env.example .env || true
echo "Edit .env with MONGO_URI, GOOGLE_API_KEY, and Endee settings, then run: python app.py"
