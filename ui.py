import os
import time
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Set aesthetic appearance modes
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MoodTrackerUI(ctk.CTk):
    """
    MoodTrackerUI builds a dark-themed AI analytics dashboard.
    Features:
      - Clean modern grid system.
      - Real-time Matplotlib emotion timeline.
      - Session counters, blink meters, and smile gauge.
      - Dynamic custom overlay toggles and controls.
    """
    def __init__(self, on_close_callback, screenshot_callback, calibration_callback):
        super().__init__()
        
        self.title("AI Facial Expression & Mood Tracker Dashboard")
        self.geometry("1280x800")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Callbacks
        self.on_close_callback = on_close_callback
        self.screenshot_callback = screenshot_callback
        self.calibration_callback = calibration_callback
        
        # State indicators
        self.dark_mode = True
        self.muted = False
        self.show_mesh = True
        self.fps = 0.0
        
        # Color Palette (Dark Theme Default)
        self.bg_color = "#1E1E24"
        self.card_color = "#2A2A35"
        self.accent_color = "#3A86C8"
        self.text_color = "#FFFFFF"

        # Emotion history for graphing (Buffer of last 40 readings)
        self.graph_buffer_size = 40
        self.timeline_data = {
            "Happy": [0.0] * self.graph_buffer_size,
            "Sad": [0.0] * self.graph_buffer_size,
            "Angry": [0.0] * self.graph_buffer_size,
            "Stressed": [0.0] * self.graph_buffer_size,
            "Neutral": [0.0] * self.graph_buffer_size
        }
        self.timeline_timestamps = list(range(self.graph_buffer_size))

        self._build_layout()

    def _build_layout(self):
        # Configure Grid Weights
        self.grid_columnconfigure(0, weight=1)  # Left Sidebar
        self.grid_columnconfigure(1, weight=4)  # Main Content (Video + Graph)
        self.grid_rowconfigure(0, weight=1)

        # ------------------ LEFT SIDEBAR ------------------
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=15)
        self.sidebar.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        # Dashboard Title
        self.title_label = ctk.CTkLabel(
            self.sidebar, 
            text="👤 MOOD TRACKER", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=25)

        # --- Metrics Panel (Sleek Grid) ---
        self.stats_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.stats_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.stats_frame.grid_columnconfigure((0, 1), weight=1)

        # Stat 1: Current Emotion
        self.emotion_title = ctk.CTkLabel(self.stats_frame, text="Current Emotion", font=ctk.CTkFont(size=12))
        self.emotion_title.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.emotion_value = ctk.CTkLabel(self.stats_frame, text="Neutral", font=ctk.CTkFont(size=22, weight="bold"), text_color="#00FFA2")
        self.emotion_value.grid(row=1, column=0, padx=5, pady=2, columnspan=2, sticky="w")

        # Stat 2: Confidence Gauge
        self.conf_title = ctk.CTkLabel(self.stats_frame, text="Confidence", font=ctk.CTkFont(size=12))
        self.conf_title.grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.conf_progressbar = ctk.CTkProgressBar(self.stats_frame, width=220)
        self.conf_progressbar.grid(row=3, column=0, columnspan=2, padx=5, pady=8, sticky="w")
        self.conf_progressbar.set(1.0)
        self.conf_text = ctk.CTkLabel(self.stats_frame, text="100%", font=ctk.CTkFont(size=12, weight="bold"))
        self.conf_text.grid(row=2, column=1, padx=5, pady=2, sticky="e")

        # Separator line
        self.sep = ctk.CTkFrame(self.sidebar, height=2, fg_color="#3A3A4A")
        self.sep.grid(row=2, column=0, padx=20, pady=15, sticky="ew")

        # Interactive telemetry
        self.telemetry_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.telemetry_frame.grid(row=3, column=0, padx=20, pady=5, sticky="nsew")
        self.telemetry_frame.grid_columnconfigure((0, 1), weight=1)

        # Blinks Metric
        self.blink_icon = ctk.CTkLabel(self.telemetry_frame, text="👁️ Blinks", font=ctk.CTkFont(size=12))
        self.blink_icon.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.blink_val = ctk.CTkLabel(self.telemetry_frame, text="0", font=ctk.CTkFont(size=18, weight="bold"))
        self.blink_val.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        # Smile Metric
        self.smile_icon = ctk.CTkLabel(self.telemetry_frame, text="😊 Smile", font=ctk.CTkFont(size=12))
        self.smile_icon.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.smile_val = ctk.CTkLabel(self.telemetry_frame, text="0%", font=ctk.CTkFont(size=18, weight="bold"))
        self.smile_val.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        # Head Pose Metric
        self.pose_title = ctk.CTkLabel(self.telemetry_frame, text="🔄 Head Pose", font=ctk.CTkFont(size=12))
        self.pose_title.grid(row=2, column=0, padx=5, pady=10, sticky="w")
        self.pose_val = ctk.CTkLabel(self.telemetry_frame, text="Y: 0° | P: 0° | R: 0°", font=ctk.CTkFont(size=11), text_color="#A0A0B0")
        self.pose_val.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky="w")

        # System telemetry
        self.fps_val = ctk.CTkLabel(self.sidebar, text="System: Running | 30 FPS", font=ctk.CTkFont(size=11), text_color="#7A7A8C")
        self.fps_val.grid(row=4, column=0, padx=20, pady=15, sticky="w")

        # Controls & Utility Section
        self.controls_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.controls_frame.grid(row=5, column=0, padx=20, pady=15, sticky="ew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        # Capture Button
        self.btn_capture = ctk.CTkButton(
            self.controls_frame, 
            text="📸 Capture Screen", 
            command=self.screenshot_callback,
            fg_color="#3B5998",
            hover_color="#4C70BA"
        )
        self.btn_capture.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Toggle Face Box
        self.btn_mesh = ctk.CTkButton(
            self.controls_frame, 
            text="🔳 Hide Face Box", 
            command=self.toggle_mesh,
            fg_color="#4E5154",
            hover_color="#5D6164"
        )
        self.btn_mesh.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Audio Alert Switcher
        self.btn_mute = ctk.CTkButton(
            self.controls_frame, 
            text="🔊 Sound Alerts: ON", 
            command=self.toggle_mute,
            fg_color="#31A354",
            hover_color="#3DAE60"
        )
        self.btn_mute.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # Color Theme Switcher
        self.btn_theme = ctk.CTkButton(
            self.controls_frame, 
            text="🌗 Light Mode", 
            command=self.toggle_theme,
            fg_color="#E08B14",
            hover_color="#F29F24"
        )
        self.btn_theme.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        # Exit App
        self.btn_exit = ctk.CTkButton(
            self.sidebar, 
            text="🚪 Save & Quit", 
            command=self.on_closing,
            fg_color="#D9534F",
            hover_color="#C9302C"
        )
        self.btn_exit.grid(row=6, column=0, padx=20, pady=25, sticky="ew")

        # ------------------ RIGHT MAIN SECTION ------------------
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=5) # Video Feed
        self.main_container.grid_rowconfigure(1, weight=3) # Matplotlib Plot

        # Video Box
        self.video_frame = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.video_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)

        # Video Label
        self.video_label = tk.Label(self.video_frame, bg="#0F0F12", borderwidth=0)
        self.video_label.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        # Analytics Plot Box
        self.plot_frame = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.plot_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        self._init_matplotlib_plot()

    def _init_matplotlib_plot(self):
        """Builds a beautiful embedded matplotlib chart styled for dark mode."""
        self.fig = Figure(figsize=(7, 2.2), dpi=100)
        self.fig.patch.set_facecolor("#2A2A35")
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1E1E24")
        
        # Grid and axes styling
        self.ax.spines['bottom'].set_color('#5A5A6A')
        self.ax.spines['left'].set_color('#5A5A6A')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        self.ax.xaxis.label.set_color('#A0A0B0')
        self.ax.yaxis.label.set_color('#A0A0B0')
        self.ax.tick_params(colors='#A0A0B0', labelsize=8)
        
        self.ax.grid(True, color="#2D2D3B", linestyle="--", linewidth=0.5)
        self.ax.set_title("Real-time Mood History Timeline", color="#FFFFFF", fontsize=10, pad=8)
        self.ax.set_ylabel("Intensity %", fontsize=8)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, self.graph_buffer_size - 1)

        # Pre-create lines with high-contrast palette
        # Cyan: Happy, Purple: Sad, Red: Angry, Orange: Stressed, Green: Neutral
        self.lines = {
            "Happy": self.ax.plot([], [], label="Happy", color="#00FFA2", linewidth=1.8)[0],
            "Sad": self.ax.plot([], [], label="Sad", color="#A27DFF", linewidth=1.5)[0],
            "Angry": self.ax.plot([], [], label="Angry", color="#FF4D4D", linewidth=1.5)[0],
            "Stressed": self.ax.plot([], [], label="Stressed", color="#FFAE42", linewidth=1.5)[0],
            "Neutral": self.ax.plot([], [], label="Neutral", color="#4DA6FF", linewidth=1.2)[0]
        }
        
        self.ax.legend(
            loc="upper right", 
            facecolor="#2A2A35", 
            edgecolor="none", 
            labelcolor="#A0A0B0",
            fontsize=7
        )

        # Place inside tkinter frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.canvas.draw()

    def update_frame(self, frame):
        """Sets the Tkinter label image dynamically with the camera frame."""
        # Convert OpenCV BGR to Pillow RGB
        cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        
        # Calculate aspect ratio to fit the label bounds nicely
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        
        # Fallback values if container hasn't fully rendered
        if label_w < 50 or label_h < 50:
            label_w, label_h = 750, 420
            
        # Standard letterboxing calculations
        img_w, img_h = pil_img.size
        ratio_w = label_w / img_w
        ratio_h = label_h / img_h
        ratio = min(ratio_w, ratio_h)
        
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Place inside photo image
        tk_img = ImageTk.PhotoImage(resized_img)
        self.video_label.configure(image=tk_img)
        self.video_label.image = tk_img

    def update_dashboard(self, emotion_result, face_metrics, fps):
        """Updates numerical metrics, telemetry texts and sliders."""
        self.fps = fps
        dominant = emotion_result.get("dominant_emotion", "Neutral")
        confidence = emotion_result.get("confidence", 0.0)

        # Dynamic Status Colors
        status_colors = {
            "Happy": "#00FFA2",
            "Sad": "#A27DFF",
            "Crying": "#8B30FF",
            "Angry": "#FF4D4D",
            "Stressed": "#FFAE42",
            "Confused": "#FFD34E",
            "Neutral": "#4DA6FF",
            "Fear": "#E080FF",
            "Disgust": "#84E296",
            "Surprise": "#FFE97D"
        }
        
        text_col = status_colors.get(dominant, "#FFFFFF")

        # 1. Update Core Stats
        self.emotion_value.configure(text=f"{dominant}", text_color=text_col)
        self.conf_text.configure(text=f"{round(confidence, 1)}%")
        self.conf_progressbar.set(confidence / 100.0)

        # 2. Update Telemetry
        blink = face_metrics.get("blink_count", 0)
        smile = face_metrics.get("smile_intensity", 0.0)
        self.blink_val.configure(text=f"{blink}")
        self.smile_val.configure(text=f"{round(smile, 1)}%")

        # Head Pose info
        p = round(face_metrics.get("pitch", 0.0), 1)
        y = round(face_metrics.get("yaw", 0.0), 1)
        r = round(face_metrics.get("roll", 0.0), 1)
        self.pose_val.configure(text=f"Yaw: {y}° | Pitch: {p}° | Roll: {r}°")

        # FPS
        self.fps_val.configure(text=f"System: Tracking Active | {round(fps, 1)} FPS")

        # 3. Stream to Matplotlib Graph
        # Extract basic and synthesized scores
        em = emotion_result.get("emotions", {})
        synth = emotion_result.get("synthesized_states", {})

        # Append new values to history buffers
        self._append_graph_point("Happy", em.get("happy", 0.0))
        self._append_graph_point("Sad", em.get("sad", 0.0))
        self._append_graph_point("Angry", em.get("angry", 0.0))
        self._append_graph_point("Stressed", synth.get("stressed", 0.0))
        self._append_graph_point("Neutral", em.get("neutral", 0.0))

        # Redraw chart
        self._redraw_plot()

    def _append_graph_point(self, name, val):
        self.timeline_data[name].append(float(val))
        if len(self.timeline_data[name]) > self.graph_buffer_size:
            self.timeline_data[name].pop(0)

    def _redraw_plot(self):
        """Re-render data lines on the embedded matplotlib graph."""
        try:
            x_vals = range(len(self.timeline_data["Happy"]))
            for name, line in self.lines.items():
                line.set_data(x_vals, self.timeline_data[name])
            
            # Rescale X axis bounds dynamically to match length
            self.ax.set_xlim(0, max(1, len(x_vals) - 1))
            self.canvas.draw_idle() # thread-safe repaint
        except Exception:
            pass

    def toggle_mesh(self):
        self.show_mesh = not self.show_mesh
        if self.show_mesh:
            self.btn_mesh.configure(text="🔳 Hide Face Box", fg_color="#4E5154", hover_color="#5D6164")
        else:
            self.btn_mesh.configure(text="🔳 Show Face Box", fg_color="#3A86C8", hover_color="#4CA0E0")

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.btn_mute.configure(text="🔇 Sound Alerts: OFF", fg_color="#4E5154", hover_color="#5D6164")
        else:
            self.btn_mute.configure(text="🔊 Sound Alerts: ON", fg_color="#31A354", hover_color="#3DAE60")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            ctk.set_appearance_mode("Dark")
            self.btn_theme.configure(text="🌗 Light Mode", fg_color="#E08B14", hover_color="#F29F24")
            self.fig.patch.set_facecolor("#2A2A35")
            self.ax.set_facecolor("#1E1E24")
            self.ax.spines['bottom'].set_color('#5A5A6A')
            self.ax.spines['left'].set_color('#5A5A6A')
            self.ax.tick_params(colors='#A0A0B0')
            self.ax.grid(True, color="#2D2D3B")
        else:
            ctk.set_appearance_mode("Light")
            self.btn_theme.configure(text="🌗 Dark Mode", fg_color="#1E1E24", hover_color="#2A2A35")
            self.fig.patch.set_facecolor("#EBEBEB")
            self.ax.set_facecolor("#FFFFFF")
            self.ax.spines['bottom'].set_color('#8A8A9A')
            self.ax.spines['left'].set_color('#8A8A9A')
            self.ax.tick_params(colors='#2A2A35')
            self.ax.grid(True, color="#E0E0E0")
        
        self.canvas.draw()

    def on_closing(self):
        """Invoke cleanup callback upon window close."""
        self.on_close_callback()
