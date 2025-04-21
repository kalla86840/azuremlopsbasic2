#!/bin/bash
echo "📦 Installing Python dependencies..."
python --version
pip install -r dev/requirements.txt
echo "✅ Done."
