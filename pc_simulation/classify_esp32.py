import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import threading
import tkinter as tk
from tkinter import font as tkfont

import torch
import requests
from PIL import Image, ImageTk
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor

# Windows: fix blurry UI on high-DPI screens
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

# ── Settings ─────────────────────────────────────────────
ESP32_IP             = "192.168.x.x"   # <-- change to your ESP32-CAM IP
CLASSIFY_EVERY       = 3.0
FETCH_EVERY          = 1.0
CONFIDENCE_THRESHOLD = 0.60
MODEL_NAME           = "openai/clip-vit-base-patch32"
WINDOW_W             = 900
CAM_W, CAM_H         = 640, 480
PANEL_W              = WINDOW_W - CAM_W
# ─────────────────────────────────────────────────────────

CAPTURE_URL = f"http://{ESP32_IP}/capture"

ALL_CLASSES = ["organic", "non_organic", "other"]

CLASS_NAMES = {
    "organic":     "Organic",
    "non_organic": "Non-Organic",
    "other":       "Other",
}
CLASS_COLORS = {
    "organic":     "#4CAF50",
    "non_organic": "#2196F3",
    "other":       "#FF9800",
}

CLIP_LABELS = [
    "organic waste such as food scraps, fruit peels, vegetable waste, rotten food, leaves, plant material",
    "non-organic waste such as plastic bottle, metal can, glass bottle, cardboard box, paper, plastic bag",
    "other waste such as electronic waste, battery, light bulb, hazardous material, mixed garbage",
]


def fetch_image():
    resp = requests.get(CAPTURE_URL, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def classify_clip(model, processor, image):
    inputs = processor(text=CLIP_LABELS, images=image,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    return [{"label": cls, "score": float(p)}
            for cls, p in zip(ALL_CLASSES, probs)]


class WasteClassifierApp:
    def __init__(self, root, model, processor):
        self.root      = root
        self.model     = model
        self.processor = processor
        self.running   = True

        root.title("Waste Classifier — ESP32-CAM")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)

        self.cam_canvas = tk.Canvas(root, width=CAM_W, height=CAM_H,
                                    bg="black", highlightthickness=0)
        self.cam_canvas.grid(row=0, column=0)

        panel = tk.Frame(root, width=PANEL_W, height=CAM_H, bg="#16213e")
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)

        title_f = tkfont.Font(family="Arial", size=11, weight="bold")
        small_f = tkfont.Font(family="Arial", size=9)

        tk.Label(panel, text="Waste Classifier", font=title_f,
                 fg="#00d4ff", bg="#16213e").pack(pady=(16, 2))
        tk.Label(panel, text=f"ESP32-CAM  {ESP32_IP}", font=small_f,
                 fg="#666688", bg="#16213e").pack()

        tk.Frame(panel, height=1, bg="#0f3460").pack(fill="x", padx=14, pady=10)

        tk.Label(panel, text="Result", font=small_f,
                 fg="#888888", bg="#16213e").pack(anchor="w", padx=16)
        self.lbl_result = tk.Label(panel, text="—", font=title_f,
                                   fg="#00d4ff", bg="#16213e")
        self.lbl_result.pack(anchor="w", padx=16)
        self.lbl_conf = tk.Label(panel, text="", font=small_f,
                                 fg="#aaaaaa", bg="#16213e")
        self.lbl_conf.pack(anchor="w", padx=16, pady=(0, 10))

        tk.Frame(panel, height=1, bg="#0f3460").pack(fill="x", padx=14, pady=(0, 10))

        tk.Label(panel, text="All classes", font=small_f,
                 fg="#888888", bg="#16213e").pack(anchor="w", padx=16)

        self.bar_vars = {}
        self.bar_lbls = {}
        bar_frame = tk.Frame(panel, bg="#16213e")
        bar_frame.pack(fill="x", padx=16, pady=6)

        BAR_MAX = PANEL_W - 80
        for cls in ALL_CLASSES:
            row = tk.Frame(bar_frame, bg="#16213e")
            row.pack(fill="x", pady=5)
            tk.Label(row, text=CLASS_NAMES[cls], font=small_f, width=11,
                     anchor="e", fg="#cccccc", bg="#16213e").pack(side="left")
            track = tk.Canvas(row, width=BAR_MAX, height=16,
                              bg="#0f3460", highlightthickness=0)
            track.pack(side="left", padx=6)
            fill = track.create_rectangle(0, 0, 0, 16,
                                          fill=CLASS_COLORS[cls], width=0)
            self.bar_vars[cls] = (track, fill, BAR_MAX)
            pct = tk.Label(row, text="0%", font=small_f, width=5,
                           anchor="w", fg="#aaaaaa", bg="#16213e")
            pct.pack(side="left")
            self.bar_lbls[cls] = pct

        tk.Frame(panel, height=1, bg="#0f3460").pack(fill="x", padx=14, pady=10)

        self.lbl_status = tk.Label(panel, text="Connecting to ESP32-CAM...",
                                   font=small_f, fg="#666688", bg="#16213e",
                                   wraplength=PANEL_W - 20)
        self.lbl_status.pack(anchor="w", padx=16)

        tk.Label(panel, text="Press Q or close window to quit",
                 font=small_f, fg="#444466", bg="#16213e").pack(side="bottom", pady=10)

        self.last_image      = None
        self.last_fetch_time = 0.0
        self.last_class_time = 0.0

        root.bind("<q>", lambda e: self.quit())
        root.bind("<Q>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self._update()

    def _update(self):
        if not self.running:
            return

        now = time.time()

        if now - self.last_fetch_time >= FETCH_EVERY:
            self.last_fetch_time = now
            threading.Thread(target=self._fetch_and_show, daemon=True).start()

        if self.last_image and now - self.last_class_time >= CLASSIFY_EVERY:
            self.last_class_time = now
            threading.Thread(target=self._classify,
                             args=(self.last_image,), daemon=True).start()

        self.root.after(200, self._update)

    def _fetch_and_show(self):
        try:
            image = fetch_image()
            self.last_image = image
            display = image.resize((CAM_W, CAM_H))
            imtk = ImageTk.PhotoImage(image=display)
            self.root.after(0, self._show_image, imtk)
            self.root.after(0, self.lbl_status.config,
                            {"text": f"Connected  —  {time.strftime('%H:%M:%S')}"})
        except Exception as e:
            self.root.after(0, self.lbl_status.config, {"text": f"Error: {e}"})

    def _show_image(self, imtk):
        self.cam_canvas.imtk = imtk
        self.cam_canvas.create_image(0, 0, anchor="nw", image=imtk)

    def _classify(self, image):
        results = classify_clip(self.model, self.processor, image)
        self.root.after(0, self._update_ui, results)

    def _update_ui(self, results):
        if not results:
            return
        results = sorted(results, key=lambda r: r["score"], reverse=True)
        top     = results[0]

        if top["score"] >= CONFIDENCE_THRESHOLD:
            name  = CLASS_NAMES.get(top["label"], top["label"])
            color = CLASS_COLORS.get(top["label"], "#ffffff")
        else:
            name  = "Not sure"
            color = "#888888"

        self.lbl_result.config(text=name, fg=color)
        self.lbl_conf.config(text=f"{top['score']*100:.1f}% confidence")

        for r in results:
            cls = r["label"]
            if cls not in self.bar_vars:
                continue
            track, fill, bar_max = self.bar_vars[cls]
            track.coords(fill, 0, 0, int(r["score"] * bar_max), 16)
            self.bar_lbls[cls].config(text=f"{r['score']*100:.1f}%")

    def quit(self):
        self.running = False
        self.root.destroy()


def load_model():
    print("Loading CLIP model (cached after first run)...")
    model     = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model ready.\n")
    return model, processor


def main():
    print(f"Connecting to ESP32-CAM at {ESP32_IP} ...")
    try:
        fetch_image()
        print("ESP32-CAM reachable.\n")
    except Exception as e:
        print(f"Cannot reach ESP32-CAM: {e}")
        print("Check the IP address and make sure the board is on the same WiFi.")
        return

    model, processor = load_model()
    root = tk.Tk()
    WasteClassifierApp(root, model, processor)
    print("Window open. Hold a waste item in front of the ESP32-CAM.")
    print("Press Q or close the window to quit.\n")
    root.mainloop()


if __name__ == "__main__":
    main()
