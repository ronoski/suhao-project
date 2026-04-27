# Waste Classifier — PC Simulation

A desktop app that classifies waste in real time into 3 categories using
**CLIP** (zero-shot AI model by OpenAI). No training required.

Supports two camera sources:
- **Laptop webcam** — for PC simulation and testing
- **ESP32-CAM** — for the full hardware project

---

## Classification Classes

| Class | Color | Examples |
|---|---|---|
| Organic | 🟢 Green | Food scraps, fruit peels, leaves, vegetable waste |
| Non-Organic | 🔵 Blue | Plastic bottles, metal cans, glass, cardboard, paper |
| Other | 🟠 Orange | Electronics, batteries, hazardous materials |

---

## Requirements

- Python 3.8 or newer — https://www.python.org/downloads/
- A working webcam (for PC simulation mode)
- ESP32-CAM running CameraWebServer sketch (for ESP32 mode)
- Internet connection — **first run only** to download the model (~350 MB).
  After that the app works fully offline.

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

Double-click `run.bat` then select mode:
```
Select mode:
  1. PC Simulation (laptop webcam)
  2. ESP32-CAM
Enter 1 or 2:
```

### Linux / Mac

```bash
bash run.sh
```

---

## ESP32-CAM Mode

Before running, set your ESP32-CAM IP address in `classify_esp32.py`:

```python
ESP32_IP = "192.168.x.x"   # <-- change this
```

Find the IP in the Arduino Serial Monitor after flashing CameraWebServer.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |

---

## Tuning

All settings are at the top of each script:

| Setting | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.60 | Below this → shows "Not sure" |
| `CLASSIFY_EVERY` | 2.0s | How often to run the model |
| `CLIP_LABELS` | — | Plain English description of each class |

To improve accuracy for a specific item, edit the `CLIP_LABELS` descriptions
— no retraining needed.

---

## Troubleshooting

**Camera not showing / black screen**
- Make sure no other app is using the webcam
- Try changing `CAM_INDEX` in `classify_webcam.py` from `0` to `1`

**On Linux: camera permission denied**
```bash
sudo chmod 666 /dev/video0
```
Permanent fix (requires logout/login):
```bash
sudo usermod -aG video $USER
```

**Cannot reach ESP32-CAM**
- Check the IP address in `classify_esp32.py`
- Make sure laptop and ESP32-CAM are on the same WiFi network
- Open `http://<ESP32_IP>/capture` in a browser to verify it works

**Model download is slow**
The model (~350 MB) downloads only on the first run and is cached locally.
Subsequent runs load instantly — no internet connection required.

**"No file found" error when running offline without prior download**
The model must be downloaded at least once while online. Run the app once
with internet access before taking it to a venue without WiFi.

**Window does not appear on Linux (Wayland)**
```bash
GDK_BACKEND=x11 bash run.sh
```

---

## Project Structure

```
pc_simulation/
├── classify_webcam.py   — PC simulation using laptop webcam
├── classify_esp32.py    — ESP32-CAM version
├── requirements.txt     — Python dependencies
├── setup.bat            — Windows setup script
├── run.bat              — Windows launcher
├── setup.sh             — Linux/Mac setup script
└── run.sh               — Linux/Mac launcher
```

---

## How It Works

This project uses **CLIP** (Contrastive Language-Image Pretraining) by OpenAI.
Unlike traditional classifiers, CLIP understands images by comparing them to
text descriptions. This means:

- No training data needed
- Classes are defined in plain English
- Easy to adjust by editing the description text

---

## Credits

- Model: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- Framework: [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- Hardware: [ESP32-CAM CameraWebServer](https://github.com/espressif/arduino-esp32)
