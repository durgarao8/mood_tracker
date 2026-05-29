import cv2
import mediapipe as mp
import numpy as np
import math

class FaceTracker:
    """
    FaceTracker leverages MediaPipe Face Mesh to analyze facial landmarks in real time.
    It calculates various physical metrics: blink detection (Eye Aspect Ratio - EAR),
    smile intensity, head movement (Yaw, Pitch, Roll), and eyebrow asymmetry.
    """
    def __init__(self, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Blink tracking state variables
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_threshold = 0.24
        self.blink_consecutive_frames = 2
        self.blink_frame_count = 0

        # Calibration/neutral values (can be updated dynamically)
        self.neutral_smile_ratio = 0.45
        self.smile_stretch_range = 0.15
        self.neutral_brow_dist_ratio = 0.28
        self.neutral_brow_height_ratio = 0.23
        self.calibration_pending = False

    def trigger_calibration(self):
        """Flags the tracker to calibrate base ratios on the next detected face."""
        self.calibration_pending = True

    def _get_distance(self, p1, p2):
        """Calculate Euclidean distance between two 3D landmarks."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def _calculate_ear(self, landmarks, eye_indices):
        """
        Calculate the Eye Aspect Ratio (EAR) for a set of eye landmarks.
        eye_indices format: [p1_outer, p2_top1, p3_top2, p4_inner, p5_bottom1, p6_bottom2]
        """
        try:
            p1 = landmarks[eye_indices[0]]
            p2 = landmarks[eye_indices[1]]
            p3 = landmarks[eye_indices[2]]
            p4 = landmarks[eye_indices[3]]
            p5 = landmarks[eye_indices[4]]
            p6 = landmarks[eye_indices[5]]

            # Vertical distances between top and bottom eyelids
            dist_v1 = self._get_distance(p2, p6)
            dist_v2 = self._get_distance(p3, p5)

            # Horizontal distance between outer and inner eye corners
            dist_h = self._get_distance(p1, p4)

            if dist_h == 0:
                return 0.0

            # EAR formula
            ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
            return ear
        except IndexError:
            return 0.0

    def analyze_landmarks(self, frame):
        """
        Processes a single BGR image frame and extracts landmark metrics.
        Returns a dictionary of analysis metrics and the annotated frame.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        h, w, c = frame.shape
        metrics = {
            "face_detected": False,
            "blink_count": self.blink_counter,
            "ear": 0.0,
            "smile_intensity": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "eyebrow_asymmetry": 0.0,
            "jaw_openness": 0.0,
            "brow_furrow_intensity": 0.0,
            "mouth_droop": 0.0,
            "brow_raise_intensity": 0.0
        }

        annotated_frame = frame.copy()

        if results.multi_face_landmarks:
            metrics["face_detected"] = True
            face_landmarks = results.multi_face_landmarks[0]

            # Convert normalized landmarks to absolute pixel values in 3D
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append([lm.x * w, lm.y * h, lm.z * w]) # scale Z by width as an approximation

            # --- Dynamic Calibration ---
            if self.calibration_pending:
                try:
                    eye_dist = self._get_distance(landmarks[33], landmarks[263])
                    if eye_dist > 0:
                        mouth_corners_dist = self._get_distance(landmarks[61], landmarks[291])
                        self.neutral_smile_ratio = mouth_corners_dist / eye_dist
                        
                        brow_dist = self._get_distance(landmarks[107], landmarks[336])
                        self.neutral_brow_dist_ratio = brow_dist / eye_dist
                        
                        brow_height_right = self._get_distance(landmarks[107], landmarks[133])
                        brow_height_left = self._get_distance(landmarks[336], landmarks[362])
                        self.neutral_brow_height_ratio = (brow_height_right + brow_height_left) / (2.0 * eye_dist)
                        
                        self.calibration_pending = False
                        print(f"[INFO] Calibrated neutral ratios: smile={round(self.neutral_smile_ratio, 3)}, brow_dist={round(self.neutral_brow_dist_ratio, 3)}, brow_height={round(self.neutral_brow_height_ratio, 3)}")
                except Exception as e:
                    print(f"[WARNING] Dynamic calibration failed: {e}")

            # --- Brow Furrow, Droop, and Raise Metrics ---
            try:
                eye_dist = self._get_distance(landmarks[33], landmarks[263])
                if eye_dist > 0:
                    brow_dist = self._get_distance(landmarks[107], landmarks[336])
                    brow_dist_ratio = brow_dist / eye_dist
                    
                    brow_height_right = self._get_distance(landmarks[107], landmarks[133])
                    brow_height_left = self._get_distance(landmarks[336], landmarks[362])
                    brow_height_ratio = (brow_height_right + brow_height_left) / (2.0 * eye_dist)
                    
                    # 1. Brow Squeeze (eyebrows moving closer together)
                    # Increased sensitivity: full squeeze represented by 15% reduction in distance
                    squeeze = (self.neutral_brow_dist_ratio - brow_dist_ratio) / (self.neutral_brow_dist_ratio * 0.15)
                    squeeze = np.clip(squeeze, 0.0, 1.0)
                    
                    # 2. Brow Lowering (eyebrows pulling down)
                    # Increased sensitivity: full lowering represented by 15% reduction in height
                    lowering = (self.neutral_brow_height_ratio - brow_height_ratio) / (self.neutral_brow_height_ratio * 0.15)
                    lowering = np.clip(lowering, 0.0, 1.0)
                    
                    # Combine squeeze and lowering: squeeze is 60%, lowering is 40%
                    brow_furrow_intensity = (squeeze * 0.6 + lowering * 0.4) * 100.0
                    metrics["brow_furrow_intensity"] = brow_furrow_intensity

                    # 3. Mouth Droop (mouth corners pulled down/in - sad/frown expression)
                    mouth_droop = 0.0
                    mouth_corners_dist = self._get_distance(landmarks[61], landmarks[291])
                    smile_ratio = mouth_corners_dist / eye_dist
                    if smile_ratio < self.neutral_smile_ratio:
                        # Full droop corresponds to 12% reduction in mouth width relative to neutral
                        droop_val = (self.neutral_smile_ratio - smile_ratio) / (self.neutral_smile_ratio * 0.12)
                        mouth_droop = np.clip(droop_val, 0.0, 1.0) * 100.0
                    metrics["mouth_droop"] = mouth_droop

                    # 4. Brow Raise (inner brows pulled up - grief/sad expression)
                    brow_raise = 0.0
                    if brow_height_ratio > self.neutral_brow_height_ratio:
                        # Full raise corresponds to 12% increase in vertical height relative to neutral
                        raise_val = (brow_height_ratio - self.neutral_brow_height_ratio) / (self.neutral_brow_height_ratio * 0.12)
                        brow_raise = np.clip(raise_val, 0.0, 1.0) * 100.0
                    metrics["brow_raise_intensity"] = brow_raise
            except IndexError:
                pass

            # --- 1. Blink Detection & EAR ---
            # Landmark indices for eyes
            # Left Eye: Outer (263), Top-left (385), Top-right (387), Inner (362), Bottom-right (373), Bottom-left (380)
            left_eye_indices = [263, 385, 387, 362, 373, 380]
            # Right Eye: Outer (33), Top-right (160), Top-left (158), Inner (133), Bottom-left (144), Bottom-right (153)
            right_eye_indices = [33, 160, 158, 133, 144, 153]

            ear_left = self._calculate_ear(landmarks, left_eye_indices)
            ear_right = self._calculate_ear(landmarks, right_eye_indices)
            avg_ear = (ear_left + ear_right) / 2.0
            metrics["ear"] = avg_ear

            # State machine for blink counting
            if avg_ear < self.blink_threshold:
                self.blink_frame_count += 1
                if self.blink_frame_count >= self.blink_consecutive_frames and not self.is_blinking:
                    self.is_blinking = True
            else:
                if self.is_blinking:
                    self.blink_counter += 1
                    self.is_blinking = False
                self.blink_frame_count = 0

            metrics["blink_count"] = self.blink_counter

            # --- 2. Smile Intensity ---
            # Mouth outer corners: 61 (right), 291 (left)
            # Stable reference: distance between outer eye corners: 33 (right eye outer), 263 (left eye outer)
            try:
                mouth_corners_dist = self._get_distance(landmarks[61], landmarks[291])
                eye_dist = self._get_distance(landmarks[33], landmarks[263])

                if eye_dist > 0:
                    smile_ratio = mouth_corners_dist / eye_dist
                    # Map to a clean percentage (0 - 100)
                    smile_val = (smile_ratio - self.neutral_smile_ratio) / self.smile_stretch_range
                    smile_intensity = np.clip(smile_val, 0.0, 1.0) * 100.0
                    metrics["smile_intensity"] = smile_intensity
            except IndexError:
                pass

            # --- 3. Eyebrow Asymmetry (Confusion Proxy) ---
            # Left eyebrow center landmark: 285 (or 300). Left eye center: 263.
            # Right eyebrow center landmark: 55 (or 70). Right eye center: 33.
            try:
                left_brow_to_eye = self._get_distance(landmarks[285], landmarks[263])
                right_brow_to_eye = self._get_distance(landmarks[55], landmarks[33])

                if right_brow_to_eye > 0:
                    ratio = left_brow_to_eye / right_brow_to_eye
                    # Eyebrow asymmetry is defined as deviation from perfect symmetry (ratio = 1.0)
                    metrics["eyebrow_asymmetry"] = abs(1.0 - ratio) * 100.0
            except IndexError:
                pass

            # --- 4. Jaw Openness ---
            # Top lip: 13, Bottom lip: 14. Normalised by nose tip (4) to chin (152) distance.
            try:
                lip_distance = self._get_distance(landmarks[13], landmarks[14])
                face_height = self._get_distance(landmarks[4], landmarks[152])
                if face_height > 0:
                    metrics["jaw_openness"] = np.clip(lip_distance / face_height, 0.0, 1.0) * 100.0
            except IndexError:
                pass

            # --- 5. Head Pose (Yaw, Pitch, Roll) ---
            # 3D points of key landmarks for projection/maths
            # 4: Nose tip, 152: Chin, 263: Left eye outer corner, 33: Right eye outer corner,
            # 291: Left mouth corner, 61: Right mouth corner
            try:
                nose_tip = landmarks[4]
                chin = landmarks[152]
                left_eye = landmarks[263]
                right_eye = landmarks[33]
                left_mouth = landmarks[291]
                right_mouth = landmarks[61]

                # Roll: Angle of eye-line with horizontal axis
                eye_dx = left_eye[0] - right_eye[0]
                eye_dy = left_eye[1] - right_eye[1]
                roll = math.degrees(math.atan2(eye_dy, eye_dx))

                # Pitch: Check position of nose tip along the vertical axis of the face (forehead to chin)
                # Forehead: landmark 10
                forehead = landmarks[10]
                total_face_v = self._get_distance(forehead, chin)
                if total_face_v > 0:
                    # Projection of nose along line forehead -> chin
                    v_vector = [chin[0] - forehead[0], chin[1] - forehead[1]]
                    v_len = math.sqrt(v_vector[0]**2 + v_vector[1]**2)
                    v_unit = [v_vector[0] / v_len, v_vector[1] / v_len]
                    
                    nose_vector = [nose_tip[0] - forehead[0], nose_tip[1] - forehead[1]]
                    proj_length = nose_vector[0] * v_unit[0] + nose_vector[1] * v_unit[1]
                    ratio = proj_length / total_face_v
                    
                    # Ideal neutral ratio is around 0.5. Scale deviation to degrees.
                    pitch = (ratio - 0.45) * 90.0 # simple linear scaling
                    metrics["pitch"] = np.clip(pitch, -45.0, 45.0)

                # Yaw: Check horizontal position of nose tip between cheek boundaries (234: Right, 454: Left)
                left_cheek = landmarks[454]
                right_cheek = landmarks[234]
                total_face_h = self._get_distance(left_cheek, right_cheek)
                if total_face_h > 0:
                    h_vector = [left_cheek[0] - right_cheek[0], left_cheek[1] - right_cheek[1]]
                    h_len = math.sqrt(h_vector[0]**2 + h_vector[1]**2)
                    h_unit = [h_vector[0] / h_len, h_vector[1] / h_len]
                    
                    nose_vector_h = [nose_tip[0] - right_cheek[0], nose_tip[1] - right_cheek[1]]
                    proj_length_h = nose_vector_h[0] * h_unit[0] + nose_vector_h[1] * h_unit[1]
                    ratio_h = proj_length_h / total_face_h
                    
                    # Ideal neutral is 0.5.
                    yaw = (ratio_h - 0.5) * 120.0
                    metrics["yaw"] = np.clip(yaw, -60.0, 60.0)

                metrics["roll"] = roll

            except (IndexError, ZeroDivisionError):
                pass

            # --- Draw Premium Bounding Box instead of Dense Mesh ---
            x_coords = [lm[0] for lm in landmarks]
            y_coords = [lm[1] for lm in landmarks]
            xmin, xmax = int(min(x_coords)), int(max(x_coords))
            ymin, ymax = int(min(y_coords)), int(max(y_coords))

            # Add padding to make the box look nice
            pad_x = int((xmax - xmin) * 0.1)
            pad_y = int((ymax - ymin) * 0.1)
            xmin = max(0, xmin - pad_x)
            xmax = min(w, xmax + pad_x)
            ymin = max(0, ymin - pad_y)
            ymax = min(h, ymax + pad_y)

            metrics["bbox"] = (xmin, ymin, xmax, ymax)

            # Draw a sleek, high-tech bounding box with corner brackets
            color = (162, 255, 0) # Neon green/cyan in BGR (#00FFA2 in RGB -> 162, 255, 0 in BGR)
            
            # Draw thin outer box
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), color, 1)

            # Draw thick corner brackets
            len_w = int((xmax - xmin) * 0.15)
            len_h = int((ymax - ymin) * 0.15)

            # Top-Left
            cv2.line(annotated_frame, (xmin, ymin), (xmin + len_w, ymin), color, 4)
            cv2.line(annotated_frame, (xmin, ymin), (xmin, ymin + len_h), color, 4)

            # Top-Right
            cv2.line(annotated_frame, (xmax, ymin), (xmax - len_w, ymin), color, 4)
            cv2.line(annotated_frame, (xmax, ymin), (xmax, ymin + len_h), color, 4)

            # Bottom-Left
            cv2.line(annotated_frame, (xmin, ymax), (xmin + len_w, ymax), color, 4)
            cv2.line(annotated_frame, (xmin, ymax), (xmin, ymax - len_h), color, 4)

            # Bottom-Right
            cv2.line(annotated_frame, (xmax, ymax), (xmax - len_w, ymax), color, 4)
            cv2.line(annotated_frame, (xmax, ymax), (xmax, ymax - len_h), color, 4)

        return metrics, annotated_frame

    def reset_blinks(self):
        """Reset the blink counter."""
        self.blink_counter = 0
