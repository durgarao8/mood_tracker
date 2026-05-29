import os
import cv2
import time
import queue
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

from face_tracker import FaceTracker
from emotion_detector import EmotionDetector
from ui import MoodTrackerUI
from utils import SoundAlertManager, CSVLogger, SessionSummary

class MoodTrackerApp:
    """
    MoodTrackerApp is the orchestrator of the entire system.
    It manages the high-performance multithreaded capture,
    threading synchronization, logging, alarms, and UI loop.
    """
    def __init__(self):
        # 1. State and thread coordination variables
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()
        
        # Performance metrics
        self.fps_timer = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0
        
        # Audio alarm states
        self.consecutive_alert_frames = 0
        self.alert_threshold_frames = 15 # alert plays after ~1.5s of continuous anger/stress
        
        # 2. Component Initializations
        self.tracker = FaceTracker()
        self.detector = EmotionDetector()
        
        # Utilities
        self.sound_manager = SoundAlertManager()
        self.logger = CSVLogger()
        self.session_summary = SessionSummary()

        # Thread-safe buffer for latest emotion analysis
        self.latest_emotion_result = {
            "dominant_emotion": "Analyzing...",
            "confidence": 0.0,
            "emotions": {"happy": 0.0, "sad": 0.0, "angry": 0.0, "fear": 0.0, "surprise": 0.0, "neutral": 100.0, "disgust": 0.0},
            "synthesized_states": {"crying": 0.0, "confused": 0.0, "stressed": 0.0}
        }

        # 3. Initialize Camera Input safely
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Try index 1 if 0 is blocked or unavailable
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("[CRITICAL] Webcam could not be opened. Fallback to offline/image validation mode.")
                messagebox.showerror(
                    "Webcam Error", 
                    "Could not access any local camera devices.\nPlease check if the webcam is connected or used by another program."
                )

        # Set standard capture properties for smoothness
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 4. Launch Background Inference Thread for DeepFace
        self.inference_thread = threading.Thread(target=self._background_inference_loop, daemon=True)
        self.inference_thread.start()

        # 5. Initialize UI Dashboard
        self.ui = MoodTrackerUI(
            on_close_callback=self.quit_app,
            screenshot_callback=self.capture_screenshot,
            calibration_callback=self.calibrate_neutral
        )

        # Periodical telemetry logging and updating
        self.last_csv_log_time = 0
        self.csv_log_interval = 1.0 # log to CSV once per second for efficiency

        # Start standard Tkinter polling loop for camera stream
        self.ui.after(10, self._camera_update_loop)

    def _background_inference_loop(self):
        """
        Background worker thread: Pulls scaled down frames, runs DeepFace model,
        and pushes result dictionary to the result queue.
        This prevents TensorFlow CPU bottlenecking the main 30 FPS camera rendering.
        """
        while self.running:
            try:
                # Blocks with timeout to avoid high CPU spin when idle
                frame_data = self.frame_queue.get(timeout=0.1)
                if frame_data is None:
                    continue

                frame, face_metrics = frame_data
                
                # Execute DeepFace inference
                analysis = self.detector.detect_emotion(frame, face_metrics)
                
                # Store result back into shared thread buffer
                self.latest_emotion_result = analysis
                
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Inference loop encountered exception: {e}")

    def _camera_update_loop(self):
        """Main rendering loop: runs on UI thread, reads camera frame, draws face mesh ~30+ FPS."""
        if not self.running:
            return

        success, frame = self.cap.read()
        if not success:
            # Generate a blank frame if camera reading drops momentarily
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Webcam feed offline or lost...", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mirror frame horizontally for intuitive desktop experience
        frame = cv2.flip(frame, 1)

        # Run high-performance MediaPipe Face Mesh on the frame
        face_metrics, annotated_frame = self.tracker.analyze_landmarks(frame)

        # Determine which frame to draw on based on mesh toggle in UI
        display_frame = annotated_frame if self.ui.show_mesh else frame

        # Synchronize sound alerts mute status
        self.sound_manager.muted = self.ui.muted

        # Extract latest emotion result
        emotion_res = self.latest_emotion_result
        dominant = emotion_res.get("dominant_emotion", "Neutral")
        confidence = emotion_res.get("confidence", 0.0)

        # --- Dynamic Text Overlay on Webcam Feed ---
        if face_metrics["face_detected"]:
            # Draw professional UI bounds directly on OpenCV canvas
            text_str = f"Emotion: {dominant} ({round(confidence, 1)}%)"
            smile_str = f"Smile: {round(face_metrics['smile_intensity'], 1)}%"
            blink_str = f"Blinks: {face_metrics['blink_count']}"
            
            # Use dynamic high-contrast colored text above the face
            cv2.putText(display_frame, text_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 162), 2, cv2.LINE_AA)
            cv2.putText(display_frame, smile_str, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, blink_str, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw premium glowing corner face tag directly above the face box
            if "bbox" in face_metrics and self.ui.show_mesh:
                xmin, ymin, xmax, ymax = face_metrics["bbox"]
                tag_text = f"{dominant} ({round(confidence, 0)}%)"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tag_ymin = max(th + 10, ymin - 8)
                # Background rect in Neon green
                cv2.rectangle(display_frame, (xmin, tag_ymin - th - 6), (xmin + tw + 10, tag_ymin), (162, 255, 0), cv2.FILLED)
                # Text in deep BGR (almost black)
                cv2.putText(display_frame, tag_text, (xmin + 5, tag_ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 30, 10), 2, cv2.LINE_AA)
            
            # --- Dynamic Queueing for background emotion inference ---
            if self.frame_queue.empty():
                # Scale down frame to speed up DeepFace processing
                small_frame = cv2.resize(frame, (320, 240))
                try:
                    self.frame_queue.put_nowait((small_frame, face_metrics))
                except queue.Full:
                    pass

            # --- Heuristic Sound Alerts Engine ---
            if dominant in ["Stressed", "Angry", "Crying"]:
                self.consecutive_alert_frames += 1
                if self.consecutive_alert_frames >= self.alert_threshold_frames:
                    if dominant == "Angry":
                        self.sound_manager.play_anger_alert()
                    else:
                        self.sound_manager.play_stress_alert()
            else:
                self.consecutive_alert_frames = max(0, self.consecutive_alert_frames - 1)

            # --- Stream CSV log and summaries periodically ---
            now = time.time()
            if now - self.last_csv_log_time >= self.csv_log_interval:
                self.logger.log_entry(emotion_res, face_metrics)
                self.session_summary.add_frame_data(
                    dominant_emotion=dominant,
                    smile_intensity=face_metrics["smile_intensity"],
                    blink_count=face_metrics["blink_count"],
                    jaw_open=face_metrics["jaw_openness"]
                )
                self.last_csv_log_time = now

        else:
            cv2.putText(display_frame, "No face detected in viewport", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

        # Update Frame Rate counters
        self.fps_counter += 1
        curr_time = time.time()
        if curr_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (curr_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = curr_time

        # Update UI components
        self.ui.update_frame(display_frame)
        self.ui.update_dashboard(emotion_res, face_metrics, self.current_fps)

        # Enqueue next camera poll
        self.ui.after(20, self._camera_update_loop)

    def capture_screenshot(self):
        """Grabs current camera frame and saves it to local disk."""
        success, frame = self.cap.read()
        if success:
            # Create folder if missing
            screenshot_dir = "screenshots"
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshot_dir, f"capture_{timestamp}.jpg")
            
            # Draw standard watermark info
            wm_frame = frame.copy()
            dominant = self.latest_emotion_result.get("dominant_emotion", "Neutral")
            text = f"Mood Tracker - Detected: {dominant} | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            cv2.putText(wm_frame, text, (15, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.imwrite(filename, wm_frame)
            messagebox.showinfo("Screenshot Captured", f"Webcam capture saved successfully to:\n{filename}")
        else:
            messagebox.showerror("Error", "Could not capture frame. Is the camera running?")

    def calibrate_neutral(self):
        """Option to dynamically calibrate base landmark distance indices (future expansion)."""
        self.tracker.trigger_calibration()
        self.tracker.reset_blinks()
        messagebox.showinfo("System Recalibrated", "Facial metrics and blink counters have been reset successfully.")

    def run(self):
        """Starts the CustomTkinter GUI main execution loop."""
        self.ui.mainloop()

    def quit_app(self):
        """Gracefully release camera resources and dump session reports upon exiting."""
        if messagebox.askokcancel("Quit Application", "Are you sure you want to save your progress and quit?"):
            self.running = False
            
            # Give background thread time to notice running=False
            time.sleep(0.2)
            
            # Release Camera safely
            if self.cap.isOpened():
                self.cap.release()
                
            # Compile and export session statistics report
            report_file = self.session_summary.generate_report()
            
            # Notify user where their reports are saved
            messagebox.showinfo(
                "Session Summary Saved",
                f"Thank you for using the AI Mood Tracker!\n\n"
                f"Your session log has been saved to CSV.\n"
                f"Your analytical report has been compiled successfully to:\n{report_file}"
            )
            
            # Destroy TK context
            self.ui.destroy()

if __name__ == "__main__":
    app = MoodTrackerApp()
    app.run()
