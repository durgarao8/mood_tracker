import cv2
import numpy as np
import time

# DeepFace can take a few seconds to import due to TensorFlow.
# We will import it safely inside the class initialization to avoid freezing the startup or main thread.
class EmotionDetector:
    """
    EmotionDetector uses DeepFace to analyze raw facial expressions from video frames.
    It runs inference in a thread-safe manner and combines DeepFace deep learning outputs
    with high-fidelity facial landmark metrics to synthesize complex states:
    Crying, Confused, and Stressed (Jealousy-like).
    """
    def __init__(self):
        self.model_loaded = False
        self.deepface_available = False
        self._initialize_model()

    def _initialize_model(self):
        """Lazy loads DeepFace and pre-warms the emotion model."""
        try:
            print("[INFO] Initializing DeepFace Emotion Model...")
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.deepface_available = True
            
            # DeepFace builds the model on first call. 
            # We pass a blank dummy image to force loading and compiling to avoid delay on first real frame.
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            self.DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
            self.model_loaded = True
            print("[INFO] DeepFace Emotion Model loaded successfully!")
        except Exception as e:
            print(f"[WARNING] DeepFace could not be initialized automatically. Error: {e}")
            print("[WARNING] The application will use a lightweight fallback emotion heuristic.")
            self.deepface_available = False
            self.model_loaded = False

    def detect_emotion(self, frame, face_metrics):
        """
        Analyzes a single frame using DeepFace.
        Merges neural network probabilities with physical landmark metrics (face_metrics)
        to return both basic and complex custom expressions.
        """
        raw_emotions = {
            "happy": 0.0, "sad": 0.0, "angry": 0.0, "fear": 0.0,
            "surprise": 0.0, "neutral": 0.0, "disgust": 0.0
        }
        dominant_emotion = "neutral"
        confidence = 0.0

        if not face_metrics.get("face_detected", False):
            return {
                "dominant_emotion": "No Face Detected",
                "confidence": 0.0,
                "emotions": raw_emotions,
                "synthesized_states": {
                    "crying": 0.0, "confused": 0.0, "stressed": 0.0
                }
            }

        # 1. Deep Learning Inference using DeepFace
        if self.deepface_available and self.model_loaded:
            try:
                # Crop face region if possible for higher accuracy, or process full frame with enforce_detection=False
                # DeepFace does automatic face detection, but we set enforce_detection=False to avoid exceptions
                results = self.DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # DeepFace returns a list of results when processing an image. Get the first face.
                if isinstance(results, list):
                    analysis = results[0]
                else:
                    analysis = results

                # Get confidence values (percentages out of 100)
                raw_emotions = {k.lower(): float(v) for k, v in analysis["emotion"].items()}
                dominant_emotion = analysis["dominant_emotion"].lower()
                confidence = raw_emotions.get(dominant_emotion, 0.0)
            except Exception as e:
                # Graceful handling of thread/resource collisions or TensorFlow exceptions
                # Fallback to landmark-based heuristics only
                pass
        
        if not self.model_loaded:
            # Landmark-based fallback: simple rule-based emotion classification
            # to make sure the app works even if DeepFace/TensorFlow fails to load
            raw_emotions, dominant_emotion, confidence = self._rule_based_fallback(face_metrics)

        # 1.5. Happy, Sad, and Angry Emotion Boosting from Physical Metrics
        # If the user shows strong physical indicators, override/boost the raw scores
        # to ensure high sensitivity even if DeepFace faces calibration or neutral bias issues.
        if face_metrics.get("face_detected", False):
            smile_intensity = face_metrics.get("smile_intensity", 0.0)
            brow_furrow = face_metrics.get("brow_furrow_intensity", 0.0)
            mouth_droop = face_metrics.get("mouth_droop", 0.0)
            brow_raise = face_metrics.get("brow_raise_intensity", 0.0)
            
            # --- Boost Happy ---
            if smile_intensity > 25.0:
                raw_emotions["happy"] = max(raw_emotions.get("happy", 0.0), smile_intensity)
            
            # --- Boost Angry ---
            if brow_furrow > 20.0:
                # Frowning represents anger/stress. If deepface reports neutral/sad but brow is furrowed, boost angry!
                raw_emotions["angry"] = max(raw_emotions.get("angry", 0.0), brow_furrow * 1.4)
                # Lower neutral so that angry dominates
                raw_emotions["neutral"] = max(0.0, raw_emotions.get("neutral", 100.0) - brow_furrow)
                
            # --- Boost Sad ---
            phys_sad = max(mouth_droop, brow_raise)
            if phys_sad > 20.0 and smile_intensity < 10.0:
                # Drooping mouth or raised inner brows represent sadness. Boost sad!
                raw_emotions["sad"] = max(raw_emotions.get("sad", 0.0), phys_sad * 1.4)
                # Lower neutral so that sad dominates
                raw_emotions["neutral"] = max(0.0, raw_emotions.get("neutral", 100.0) - phys_sad)

            # Re-determine dominant emotion and confidence based on our boosted raw values
            dominant_emotion = max(raw_emotions, key=raw_emotions.get)
            confidence = raw_emotions[dominant_emotion]

        # 2. Custom Heuristic Engine for Synthesized States
        # Combining deep learning emotion results with direct landmark ratios
        smile_intensity = face_metrics.get("smile_intensity", 0.0)
        ear = face_metrics.get("ear", 0.3)
        eyebrow_asymmetry = face_metrics.get("eyebrow_asymmetry", 0.0)
        jaw_openness = face_metrics.get("jaw_openness", 0.0)
        pitch = face_metrics.get("pitch", 0.0)
        yaw = face_metrics.get("yaw", 0.0)

        # Calculate Crying
        # Crying: High Sad probability + low smile intensity + squinting (low EAR) or frequent blinking.
        crying_prob = 0.0
        if raw_emotions.get("sad", 0.0) > 30.0:
            smile_penalty = max(0.0, 1.0 - (smile_intensity / 20.0)) # Crying face does not smile
            squint_bonus = 1.0 if ear < 0.23 else 0.5
            crying_prob = raw_emotions["sad"] * smile_penalty * squint_bonus
        crying_prob = np.clip(crying_prob, 0.0, 100.0)

        # Calculate Confused
        # Confused: Neutral/Sad active + significant eyebrow asymmetry (raised eyebrow)
        confused_prob = 0.0
        neutral_sad_base = max(raw_emotions.get("neutral", 0.0), raw_emotions.get("sad", 0.0))
        if neutral_sad_base > 25.0:
            asymmetry_bonus = np.clip(eyebrow_asymmetry / 15.0, 0.0, 1.0) # threshold at 15% deviation
            confused_prob = neutral_sad_base * asymmetry_bonus
        confused_prob = np.clip(confused_prob, 0.0, 100.0)

        # Calculate Stressed / Jealousy-like
        # Stressed: High Angry or Fear or physical brow furrowing + low EAR (glare/squint) + rigid chin (jaw openness low) 
        # Or head slightly turned away while eyes remain forward (yaw vs eye direction)
        stressed_prob = 0.0
        brow_furrow = face_metrics.get("brow_furrow_intensity", 0.0)
        
        stress_base = max(raw_emotions.get("angry", 0.0), raw_emotions.get("fear", 0.0), brow_furrow)
        if stress_base > 10.0:
            squint_bonus = 1.25 if ear < 0.23 else 0.85
            jaw_tension_bonus = 1.2 if jaw_openness < 12.0 else 0.7
            head_tilt_bonus = 1.15 if abs(yaw) > 12.0 else 1.0 # looking away slightly
            stressed_prob = stress_base * squint_bonus * jaw_tension_bonus * head_tilt_bonus
        else:
            # Frown stress: high neutral + squeezed eyebrows (furrowing)
            if raw_emotions.get("neutral", 0.0) > 40.0 and brow_furrow > 15.0:
                stressed_prob = raw_emotions["neutral"] * (brow_furrow / 30.0)

        stressed_prob = np.clip(stressed_prob, 0.0, 100.0)

        # 3. Formulate Final Dashboard Outputs
        all_emotions = {**raw_emotions}
        
        # We can dynamically inject synthesized emotions into the dominant classification
        # if their synthesized confidence exceeds the threshold and overrides the basic ones!
        final_dominant = dominant_emotion
        final_confidence = confidence

        if crying_prob > 50.0 and crying_prob > confidence:
            final_dominant = "crying"
            final_confidence = crying_prob
        elif stressed_prob > 50.0 and stressed_prob > confidence:
            final_dominant = "stressed"
            final_confidence = stressed_prob
        elif confused_prob > 50.0 and confused_prob > confidence:
            final_dominant = "confused"
            final_confidence = confused_prob

        return {
            "dominant_emotion": final_dominant.capitalize(),
            "confidence": final_confidence,
            "emotions": all_emotions,
            "synthesized_states": {
                "crying": crying_prob,
                "confused": confused_prob,
                "stressed": stressed_prob
            }
        }

    def _rule_based_fallback(self, face_metrics):
        """
        A rule-based emotion engine that runs purely on physical face metrics.
        Ensures the app functions beautifully as a backup if TensorFlow/DeepFace fails to load.
        """
        emotions = {
            "happy": 0.0, "sad": 0.0, "angry": 0.0, "fear": 0.0,
            "surprise": 0.0, "neutral": 100.0, "disgust": 0.0
        }
        
        smile = face_metrics.get("smile_intensity", 0.0)
        ear = face_metrics.get("ear", 0.3)
        jaw = face_metrics.get("jaw_openness", 0.0)
        asymmetry = face_metrics.get("eyebrow_asymmetry", 0.0)
        brow_furrow = face_metrics.get("brow_furrow_intensity", 0.0)
        mouth_droop = face_metrics.get("mouth_droop", 0.0)
        brow_raise = face_metrics.get("brow_raise_intensity", 0.0)

        # Basic landmark heuristics
        if smile > 30.0:
            emotions["happy"] = smile
            emotions["neutral"] = max(0.0, 100.0 - smile)
        elif jaw > 40.0 and ear > 0.25:
            emotions["surprise"] = min(jaw * 1.5, 100.0)
            emotions["neutral"] = max(0.0, 100.0 - emotions["surprise"])
        elif brow_furrow > 20.0 and smile < 15.0:
            # High frowning + low smile = anger
            emotions["angry"] = min(brow_furrow * 1.8, 100.0)
            emotions["neutral"] = max(0.0, 100.0 - emotions["angry"])
            # Subtle fear / worry can be blended
            if ear < 0.23:
                emotions["fear"] = min(brow_furrow * 0.8, 100.0)
        elif brow_furrow > 15.0 and ear < 0.23 and smile < 10.0:
            # Furrowed brows + squinting/tension + tight jaw = fear/stress base
            emotions["fear"] = min(brow_furrow * 1.2, 100.0)
            emotions["neutral"] = max(0.0, 100.0 - emotions["fear"])
        elif (mouth_droop > 20.0 or brow_raise > 20.0) and smile < 10.0:
            # Drooping mouth corners or raised inner eyebrows = sadness
            sad_score = max(mouth_droop, brow_raise)
            emotions["sad"] = min(sad_score * 1.8, 100.0)
            emotions["neutral"] = max(0.0, 100.0 - emotions["sad"])
        elif ear < 0.22 and smile < 10.0:
            # Squinting or sad
            emotions["sad"] = 60.0
            emotions["neutral"] = 40.0
        elif asymmetry > 12.0:
            emotions["sad"] = 30.0
            emotions["neutral"] = 70.0

        # Adjust neutral to fill remaining probability
        other_sum = sum(v for k, v in emotions.items() if k != "neutral")
        emotions["neutral"] = max(0.0, 100.0 - other_sum)

        # Select dominant
        dom = max(emotions, key=emotions.get)
        conf = emotions[dom]

        return emotions, dom, conf
