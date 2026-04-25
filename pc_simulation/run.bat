@echo off
call venv\Scripts\activate

echo Select mode:
echo   1. PC Simulation (laptop webcam)
echo   2. ESP32-CAM
set /p choice="Enter 1 or 2: "

if "%choice%"=="2" (
    python classify_esp32.py
) else (
    python classify_webcam.py
)
pause
