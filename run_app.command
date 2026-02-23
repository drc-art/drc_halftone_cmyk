#!/bin/bash
cd "$(dirname "$0")"

# Try python3 first (mac standard)
python3 "drc_halftone_cmyk.py"
if [ $? -eq 0 ]; then
  exit 0
fi

# Fallback to python
python "drc_halftone_cmyk.py"
if [ $? -eq 0 ]; then
  exit 0
fi

echo ""
echo "Failed to start app. Run setup_env.command first."
read -p "Press Enter to close..."
