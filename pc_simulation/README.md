# Waste Classifier — PC Simulation

A desktop app that uses your laptop webcam to classify waste in real time using
a pre-trained AI model. Built as a PC simulation for the ESP32-CAM Waste
Classification project.

---

## What It Does

- Opens your webcam in a window
- Every 2 seconds, takes a snapshot and runs it through an AI model
- Displays the predicted waste category and confidence for all 6 classes

**Classes:** Cardboard · Glass · Metal · Paper · Plastic · Trash

---

## Requirements

- Python 3.8 or newer — https://www.python.org/downloads/
- A working webcam
- Internet connection (first run only, to download the model ~400 MB)

---

## Setup (run once)

### Windows

1. Install Python from https://www.python.org/downloads/
   - During install, check **"Add Python to PATH"**
2. Double-click `setup.bat`
3. Wait for it to finish

### Linux / Mac

```bash
bash setup.sh
```

---

## Run

### Windows

Double-click `run.bat`

### Linux / Mac

```bash
bash run.sh
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save snapshot as `snapshot_000.jpg` |

---

## Troubleshooting

**Camera not showing / black screen**

- Make sure no other app is using the webcam (Zoom, Teams, etc.)
- Try changing `CAM_INDEX` at the top of `classify_webcam.py` from `0` to `1`

**On Linux: camera permission denied**

```bash
sudo chmod 666 /dev/video0
```

**Model download is slow**

The model (~400 MB) downloads only on the first run and is cached locally.
Subsequent runs load instantly.

**Window does not appear on Linux (Wayland)**

The app uses Tkinter which works on both X11 and Wayland. If the window
does not appear, try:

```bash
GDK_BACKEND=x11 bash run.sh
```

---

## Project Structure

```
pc_simulation/
├── classify_webcam.py   — main application
├── requirements.txt     — Python dependencies
├── setup.bat            — Windows setup script
├── run.bat              — Windows launcher
├── setup.sh             — Linux/Mac setup script
└── run.sh               — Linux/Mac launcher
```

---

## Next Steps

This simulation uses a general waste model trained on the TrashNet dataset.
For the full project with the ESP32-CAM hardware, the app can be extended to:

- Fetch images from the ESP32-CAM over WiFi instead of the laptop webcam
- Classify into **Organic**, **Non-organic**, and **Other** using a custom
  Teachable Machine model trained on your own photos
- Trigger a physical sorting mechanism (servo, LED) on the ESP32

---

## Credits

- Model: [yangy50/garbage-classification](https://huggingface.co/yangy50/garbage-classification) on Hugging Face
- Dataset: [TrashNet](https://github.com/garythung/trashnet) by Gary Thung
- Framework: [Hugging Face Transformers](https://huggingface.co/docs/transformers)
