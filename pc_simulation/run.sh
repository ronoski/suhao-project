#!/bin/bash
source venv/bin/activate

echo "Select mode:"
echo "  1. PC Simulation (laptop webcam)"
echo "  2. ESP32-CAM"
read -p "Enter 1 or 2: " choice

if [ "$choice" = "2" ]; then
    python3 classify_esp32.py
else
    python3 classify_webcam.py
fi
