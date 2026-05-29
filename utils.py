import os
import time
import csv
import numpy as np
from datetime import datetime
import pygame

class SoundAlertManager:
    """
    Dynamically synthesizes warning tones using Pygame and NumPy.
    This eliminates the need for external MP3/WAV files and ensures the sound feature
    works completely offline and out-of-the-box.
    """
    def __init__(self):
        self.muted = False
        self.last_played_time = 0
        self.cooldown_period = 3.0 # seconds between alerts

        # Initialize Pygame Mixer safely
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
            self.mixer_enabled = True
            # Generate the alert sound buffer (440Hz / A4 note warning beep)
            self.warning_sound = self._synthesize_beep(frequency=580.0, duration=0.2)
            self.stress_sound = self._synthesize_beep(frequency=400.0, duration=0.3)
        except Exception as e:
            print(f"[WARNING] Pygame mixer initialization failed: {e}. Audio alerts will be disabled.")
            self.mixer_enabled = False

    def _synthesize_beep(self, frequency, duration):
        """Generates a stereo sine wave beep dynamically as a Pygame Sound."""
        if not self.mixer_enabled:
            return None
        
        sample_rate = 44100
        num_samples = int(sample_rate * duration)
        
        # Create sine wave
        t = np.linspace(0, duration, num_samples, False)
        # Sine wave mapped to 16-bit integer range [-32768, 32767]
        wave = np.sin(2 * np.pi * frequency * t) * 32767
        wave = wave.astype(np.int16)
        
        # Convert to 2-channel (stereo) array
        stereo_wave = np.dstack((wave, wave))[0]
        
        try:
            return pygame.sndarray.make_sound(stereo_wave)
        except Exception:
            return None

    def play_anger_alert(self):
        """Plays a sharp alert tone for anger detection."""
        if self.muted or not self.mixer_enabled or not self.warning_sound:
            return
        
        now = time.time()
        if now - self.last_played_time > self.cooldown_period:
            self.warning_sound.play()
            self.last_played_time = now

    def play_stress_alert(self):
        """Plays a distinct alert tone for high stress."""
        if self.muted or not self.mixer_enabled or not self.stress_sound:
            return
        
        now = time.time()
        if now - self.last_played_time > self.cooldown_period:
            self.stress_sound.play()
            self.last_played_time = now

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted


class CSVLogger:
    """
    Handles file operations for recording emotional data streams over time.
    Saves metadata with structured timestamping.
    """
    def __init__(self, output_dir="logs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Generate log filename based on session launch timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.output_dir, f"session_log_{timestamp_str}.csv")
        self.headers = [
            "Timestamp", "Dominant_Emotion", "Confidence",
            "Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral", "Disgust",
            "Crying", "Confused", "Stressed",
            "Blink_Count", "Smile_Intensity", "Pitch", "Yaw", "Roll", "Jaw_Openness"
        ]
        
        # Initialize the CSV with headers
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_entry(self, emotion_data, face_metrics):
        """Streams a single multi-dimensional frame analysis to the CSV."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            em = emotion_data.get("emotions", {})
            synth = emotion_data.get("synthesized_states", {})

            row = [
                timestamp,
                emotion_data.get("dominant_emotion", "Neutral"),
                round(emotion_data.get("confidence", 0.0), 2),
                round(em.get("happy", 0.0), 2),
                round(em.get("sad", 0.0), 2),
                round(em.get("angry", 0.0), 2),
                round(em.get("fear", 0.0), 2),
                round(em.get("surprise", 0.0), 2),
                round(em.get("neutral", 0.0), 2),
                round(em.get("disgust", 0.0), 2),
                round(synth.get("crying", 0.0), 2),
                round(synth.get("confused", 0.0), 2),
                round(synth.get("stressed", 0.0), 2),
                face_metrics.get("blink_count", 0),
                round(face_metrics.get("smile_intensity", 0.0), 2),
                round(face_metrics.get("pitch", 0.0), 2),
                round(face_metrics.get("yaw", 0.0), 2),
                round(face_metrics.get("roll", 0.0), 2),
                round(face_metrics.get("jaw_openness", 0.0), 2)
            ]

            with open(self.filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"[ERROR] Failed to write log to CSV: {e}")


class SessionSummary:
    """
    Aggregates metrics and compile statistics to generate a beautiful, professional
    markdown or text summary report at the end of the session.
    """
    def __init__(self):
        self.start_time = time.time()
        self.emotion_history = []
        self.smile_intensities = []
        self.blink_count = 0
        self.jaw_openness_history = []
        
    def add_frame_data(self, dominant_emotion, smile_intensity, blink_count, jaw_open):
        """Accumulates real-time stats for the summary report."""
        self.emotion_history.append(dominant_emotion.capitalize())
        self.smile_intensities.append(smile_intensity)
        self.blink_count = blink_count
        self.jaw_openness_history.append(jaw_open)

    def generate_report(self, output_dir="reports"):
        """Compiles session averages and exports a clean markdown report."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        duration = time.time() - self.start_time
        duration_minutes = round(duration / 60, 2)
        total_frames = len(self.emotion_history)

        if total_frames == 0:
            return "No data recorded during session."

        # Compute percentages of emotions
        unique_emotions, counts = np.unique(self.emotion_history, return_counts=True)
        percentages = (counts / total_frames) * 100
        emotion_dist = dict(zip(unique_emotions, percentages))

        # Sort distribution by highest percentage
        sorted_dist = sorted(emotion_dist.items(), key=lambda item: item[1], reverse=True)

        avg_smile = round(np.mean(self.smile_intensities), 2) if self.smile_intensities else 0.0
        max_smile = round(np.max(self.smile_intensities), 2) if self.smile_intensities else 0.0
        avg_jaw = round(np.mean(self.jaw_openness_history), 2) if self.jaw_openness_history else 0.0

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        report_path = os.path.join(output_dir, f"session_summary_{timestamp_str}.md")

        report_content = f"""# AI Mood Tracker - Session Summary Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## ⏱️ Session Metrics
- **Total Session Duration**: {duration_minutes} minutes ({round(duration, 1)} seconds)
- **Total Frames Analyzed**: {total_frames}
- **Total Eye Blinks**: {self.blink_count}
- **Average Blink Frequency**: {round(self.blink_count / (duration_minutes or 1), 2)} blinks/min

---

## 📊 Emotional State Distribution
The percentages below indicate the proportion of the session spent in each emotional state:

| Emotion | Duration % | Visual Level |
| :--- | :--- | :--- |
"""
        for em, pct in sorted_dist:
            bar = "█" * int(pct / 5)
            report_content += f"| **{em}** | {round(pct, 1)}% | {bar} |\n"

        report_content += f"""
---

## 🧬 Facial Expression Analysis
- **Average Smile Intensity**: {avg_smile}%
- **Peak Smile Intensity**: {max_smile}%
- **Average Jaw Openness (Speech/Expression)**: {avg_jaw}%

## 💡 Key Insights
"""
        # Formulate some smart insights
        dominant_top = sorted_dist[0][0]
        if dominant_top == "Happy":
            report_content += "- **Overall Mood**: Your overall mood was predominantly positive and bright!\n"
        elif dominant_top == "Sad" or dominant_top == "Crying":
            report_content += "- **Overall Mood**: Your facial expressions showed deep distress or low energy. Consider taking a quick break to stretch and drink some water.\n"
        elif dominant_top == "Stressed":
            report_content += "- **Overall Mood**: High muscle tension, low EAR, or stress indicators were detected frequently. Try some relaxing breathing exercises.\n"
        elif dominant_top == "Angry":
            report_content += "- **Overall Mood**: Multiple flashes of frustration were registered. Remember to take deep breaths to reduce cognitive strain.\n"
        else:
            report_content += "- **Overall Mood**: You maintained a very focused, neutral, and calm composure.\n"

        if self.blink_count / (duration_minutes or 1) < 8.0:
            report_content += "- **Eye Strain Warning**: Your blink rate was lower than average (typically 12-15 per min). You might be experiencing dry eyes or screen fatigue. Remember the 20-20-20 rule!\n"

        with open(report_path, "w") as f:
            f.write(report_content)

        return report_path
