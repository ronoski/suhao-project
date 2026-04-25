import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import threading
import tkinter as tk
from tkinter import font as tkfont

import cv2
from PIL import Image, ImageTk
from transformers import pipeline

# Windows: fix blurry UI on high-DPI screens
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

# ── Settings ─────────────────────────────────────────────
CLASSIFY_EVERY = 2.0
MODEL_NAME     = "yangy50/garbage-classification"
WINDOW_W       = 900
CAM_W, CAM_H   = 640, 480
PANEL_W        = WINDOW_W - CAM_W
# ─────────────────────────────────────────────────────────

CLASS_COLORS = {
    "cardboard": "#FFA500",
    "glass":     "#00FFFF",
    "metal":     "#C0C0C0",
    "paper":     "#FFFFFF",
    "plastic":   "#32DC32",
    "trash":     "#FF4444",
}
CLASS_NAMES = {
    "cardboard": "Cardboard",
    "glass":     "Glass",
    "metal":     "Metal",
    "paper":     "Paper",
    "plastic":   "Plastic",
    "trash":     "Trash",
}
ALL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class WasteClassifierApp:
    def __init__(self, root, model, cam_index):
        self.root    = root
        self.model   = model
        self.results = []
        self.running = True

        root.title("Waste Classifier — PC Simulation")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)

        # ── left: camera canvas ───────────────────────────
        self.cam_canvas = tk.Canvas(root, width=CAM_W, height=CAM_H,
                                    bg="black", highlightthickness=0)
        self.cam_canvas.grid(row=0, column=0)

        # ── right: info panel ─────────────────────────────
        panel = tk.Frame(root, width=PANEL_W, height=CAM_H,
                         bg="#16213e")
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)

        title_f = tkfont.Font(family="Arial", size=11, weight="bold")
        small_f = tkfont.Font(family="Arial", size=9)

        tk.Label(panel, text="Waste Classifier", font=title_f,
                 fg="#00d4ff", bg="#16213e").pack(pady=(16, 2))
        tk.Label(panel, text="PC Simulation", font=small_f,
                 fg="#666688", bg="#16213e").pack()

        tk.Frame(panel, height=1, bg="#0f3460").pack(fill="x", padx=14, pady=10)

        # top result
        tk.Label(panel, text="Result", font=small_f,
                 fg="#888888", bg="#16213e").pack(anchor="w", padx=16)
        self.lbl_result = tk.Label(panel, text="—", font=title_f,
                                   fg="#00d4ff", bg="#16213e")
        self.lbl_result.pack(anchor="w", padx=16)
        self.lbl_conf = tk.Label(panel, text="", font=small_f,
                                 fg="#aaaaaa", bg="#16213e")
        self.lbl_conf.pack(anchor="w", padx=16, pady=(0, 10))

        tk.Frame(panel, height=1, bg="#0f3460").pack(fill="x", padx=14, pady=(0, 10))

        # bars
        tk.Label(panel, text="All classes", font=small_f,
                 fg="#888888", bg="#16213e").pack(anchor="w", padx=16)

        self.bar_vars  = {}
        self.bar_lbls  = {}
        bar_frame = tk.Frame(panel, bg="#16213e")
        bar_frame.pack(fill="x", padx=16, pady=6)

        BAR_MAX = PANEL_W - 80

        for cls in ALL_CLASSES:
            row = tk.Frame(bar_frame, bg="#16213e")
            row.pack(fill="x", pady=3)

            tk.Label(row, text=CLASS_NAMES[cls], font=small_f, width=9,
                     anchor="e", fg="#cccccc", bg="#16213e").pack(side="left")

            track = tk.Canvas(row, width=BAR_MAX, height=14,
                              bg="#0f3460", highlightthickness=0)
            track.pack(side="left", padx=6)

            fill = track.create_rectangle(0, 0, 0, 14,
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

        # ── camera + classify threads ──────────────────────
        backend      = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap     = cv2.VideoCapture(cam_index, backend)
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

        self.root.after(30, self._update)   # ~33 fps

    def _show_frame(self, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)
        imtk = ImageTk.PhotoImage(image=img)
        self.cam_canvas.imtk = imtk          # keep reference
        self.cam_canvas.create_image(0, 0, anchor="nw", image=imtk)

    def _classify(self, frame):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image   = Image.fromarray(rgb)
        results = self.model(image, top_k=6)
        self.root.after(0, self._update_ui, results)

    def _update_ui(self, results):
        if not results:
            return

        # sort by score descending
        results = sorted(results, key=lambda r: r["score"], reverse=True)
        top     = results[0]

        name  = CLASS_NAMES.get(top["label"], top["label"])
        color = CLASS_COLORS.get(top["label"], "#ffffff")
        self.lbl_result.config(text=name, fg=color)
        self.lbl_conf.config(text=f"{top['score']*100:.1f}% confidence")
        self.lbl_status.config(
            text=f"Last update: {time.strftime('%H:%M:%S')}")

        score_map = {r["label"]: r["score"] for r in results}
        for cls, (track, fill, bar_max) in self.bar_vars.items():
            score = score_map.get(cls, 0.0)
            w = int(score * bar_max)
            track.coords(fill, 0, 0, w, 14)
            self.bar_lbls[cls].config(text=f"{score*100:.1f}%")

    def quit(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


def find_camera():
    # Use DirectShow on Windows for faster camera probing
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
    print("Loading model (cached after first run)...")
    m = pipeline("image-classification", model=MODEL_NAME)
    print("Model ready.\n")
    return m


def main():
    cam_index = find_camera()
    if cam_index is None:
        print("No camera found. Please connect a webcam and try again.")
        return

    model = load_model()
    root  = tk.Tk()
    app   = WasteClassifierApp(root, model, cam_index)
    print("Window open. Hold a waste item in front of the camera.")
    print("Press Q or close the window to quit.\n")
    root.mainloop()


if __name__ == "__main__":
    main()
