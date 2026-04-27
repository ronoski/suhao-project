import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import threading
import tkinter as tk
from tkinter import font as tkfont

import cv2
import torch
from PIL import Image, ImageTk
from transformers import CLIPModel, CLIPProcessor

# Windows: fix blurry UI on high-DPI screens
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

# ── Settings ─────────────────────────────────────────────
CLASSIFY_EVERY       = 2.0
CONFIDENCE_THRESHOLD = 0.60   # below this → show "Not sure"
MODEL_NAME           = "openai/clip-vit-base-patch32"
WINDOW_W             = 900
CAM_W, CAM_H         = 640, 480
PANEL_W              = WINDOW_W - CAM_W
# ─────────────────────────────────────────────────────────

# Short keys used internally
ALL_CLASSES = ["organic", "non_organic", "other"]

CLASS_NAMES = {
    "organic":     "Organic",
    "non_organic": "Non-Organic",
    "other":       "Other",
}
CLASS_COLORS = {
    "organic":     "#4CAF50",   # green
    "non_organic": "#2196F3",   # blue
    "other":       "#FF9800",   # orange
}

# Descriptive text CLIP uses to understand each class
CLIP_LABELS = [
    "organic waste such as food scraps, fruit peels, vegetable waste, rotten food, leaves, plant material",
    "non-organic waste such as plastic bottle, metal can, glass bottle, cardboard box, paper, plastic bag",
    "other waste such as electronic waste, battery, light bulb, hazardous material, mixed garbage",
]


def classify_clip(model, processor, image):
    inputs = processor(text=CLIP_LABELS, images=image,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    return [{"label": cls, "score": float(p)}
            for cls, p in zip(ALL_CLASSES, probs)]


class WasteClassifierApp:
    def __init__(self, root, model, processor, cam_index):
        self.root      = root
        self.model     = model
        self.processor = processor
        self.running   = True

        root.title("Waste Classifier — PC Simulation")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)

        # ── left: camera canvas ───────────────────────────
        self.cam_canvas = tk.Canvas(root, width=CAM_W, height=CAM_H,
                                    bg="black", highlightthickness=0)
        self.cam_canvas.grid(row=0, column=0)

        # ── right: info panel ─────────────────────────────
        panel = tk.Frame(root, width=PANEL_W, height=CAM_H, bg="#16213e")
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)

        title_f = tkfont.Font(family="Arial", size=11, weight="bold")
        small_f = tkfont.Font(family="Arial", size=9)

        tk.Label(panel, text="Waste Classifier", font=title_f,
                 fg="#00d4ff", bg="#16213e").pack(pady=(16, 2))
        tk.Label(panel, text="PC Simulation", font=small_f,
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

        self.lbl_status = tk.Label(panel, text="Starting...", font=small_f,
                                   fg="#666688", bg="#16213e", wraplength=PANEL_W - 20)
        self.lbl_status.pack(anchor="w", padx=16)

        tk.Label(panel, text="Press Q or close window to quit",
                 font=small_f, fg="#444466", bg="#16213e").pack(side="bottom", pady=10)

        backend          = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap         = cv2.VideoCapture(cam_index, backend)
        self.last_frame  = None
        self.lock        = threading.Lock()
        self.last_time   = 0.0

        root.bind("<q>", lambda e: self.quit())
        root.bind("<Q>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self._update()

    def _update(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.last_frame = frame.copy()
            self._show_frame(frame)

            now = time.time()
            if now - self.last_time >= CLASSIFY_EVERY:
                self.last_time = now
                threading.Thread(target=self._classify,
                                 args=(frame,), daemon=True).start()

        self.root.after(30, self._update)

    def _show_frame(self, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)
        imtk = ImageTk.PhotoImage(image=img)
        self.cam_canvas.imtk = imtk
        self.cam_canvas.create_image(0, 0, anchor="nw", image=imtk)

    def _classify(self, frame):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image   = Image.fromarray(rgb)
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
        self.lbl_status.config(text=f"Last update: {time.strftime('%H:%M:%S')}")

        for r in results:
            cls = r["label"]
            if cls not in self.bar_vars:
                continue
            track, fill, bar_max = self.bar_vars[cls]
            track.coords(fill, 0, 0, int(r["score"] * bar_max), 16)
            self.bar_lbls[cls].config(text=f"{r['score']*100:.1f}%")

    def quit(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


def find_camera():
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for index in range(5):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Camera found at index {index}")
                return index
    return None


def load_model():
    print("Loading CLIP model (cached after first run)...")
    model     = CLIPModel.from_pretrained(MODEL_NAME, local_files_only=True)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    print("Model ready.\n")
    return model, processor


def main():
    cam_index = find_camera()
    if cam_index is None:
        print("No camera found. Please connect a webcam and try again.")
        return

    model, processor = load_model()
    root = tk.Tk()
    WasteClassifierApp(root, model, processor, cam_index)
    print("Window open. Hold a waste item in front of the camera.")
    print("Press Q or close the window to quit.\n")
    root.mainloop()


if __name__ == "__main__":
    main()
