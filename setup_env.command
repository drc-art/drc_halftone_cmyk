#!/bin/bash
cd "$(dirname "$0")"

echo "Upgrading pip..."
python3 -m pip install --upgrade pip || python -m pip install --upgrade pip

echo "Installing requirements..."
python3 -m pip install -r requirements.txt || python -m pip install -r requirements.txt

echo ""
echo "Setup complete. Use run_app.command to start the app."
read -p "Press Enter to close..."
