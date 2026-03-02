import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Try new mediapipe API first, fall back to legacy
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    _USE_LEGACY = False
except Exception:
    _USE_LEGACY = True

# Always try legacy as it's more widely supported
try:
    import mediapipe as mp
    _mp_pose    = mp.solutions.pose
    _USE_LEGACY = True
except Exception:
    _USE_LEGACY = False


class PoseAnalyzer:
    """
    MediaPipe-based pose landmark detection and angle computation.
    Supports both legacy (0.9.x) and new (0.10.x) mediapipe APIs.
    """

    LM = {
        "nose":           0,
        "l_shoulder":    11, "r_shoulder":    12,
        "l_elbow":       13, "r_elbow":       14,
        "l_wrist":       15, "r_wrist":       16,
        "l_hip":         23, "r_hip":         24,
        "l_knee":        25, "r_knee":        26,
        "l_ankle":       27, "r_ankle":       28,
        "l_heel":        29, "r_heel":        30,
        "l_foot_index":  31, "r_foot_index":  32,
    }

    def __init__(self):
        import mediapipe as mp

        # Try legacy solutions API
        try:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mode = "legacy"
        except AttributeError:
            # New tasks API
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                import urllib.request, os, tempfile

                model_url  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve(model_url, model_path)

                base_options    = python.BaseOptions(model_asset_path=model_path)
                options         = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    output_segmentation_masks=False,
                    running_mode=vision.RunningMode.VIDEO,
                )
                self._pose      = vision.PoseLandmarker.create_from_options(options)
                self._mode      = "tasks"
                self._ts        = 0
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialise MediaPipe Pose.\n"
                    f"Try: pip install mediapipe==0.10.11\nError: {e}"
                )

    # ── frame processing ─────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._mode == "legacy":
            results = self._pose.process(rgb)
            if results.pose_landmarks:
                return results.pose_landmarks.landmark, results
            return None, None

        else:  # tasks API
            self._ts += 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = self._pose.detect_for_video(mp_image, self._ts)
            if results.pose_landmarks:
                return results.pose_landmarks[0], results
            return None, None

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _lm_to_px(lm, idx: int, w: int, h: int) -> np.ndarray:
        p = lm[idx]
        return np.array([p.x * w, p.y * h])

    @staticmethod
    def _angle_3pts(a, b, c) -> float:
        ba = a - b
        bc = c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    # ── public API ───────────────────────────────────────────────────────────

    def calculate_angles(self, lm, w: int, h: int) -> Dict[str, float]:
        px = lambda idx: self._lm_to_px(lm, idx, w, h)
        L  = self.LM

        l_sh  = px(L["l_shoulder"]);  r_sh  = px(L["r_shoulder"])
        l_el  = px(L["l_elbow"]);     r_el  = px(L["r_elbow"])
        l_wr  = px(L["l_wrist"]);     r_wr  = px(L["r_wrist"])
        l_hip = px(L["l_hip"]);       r_hip = px(L["r_hip"])
        l_kn  = px(L["l_knee"]);      r_kn  = px(L["r_knee"])
        l_an  = px(L["l_ankle"]);     r_an  = px(L["r_ankle"])
        nose  = px(L["nose"])

        mid_sh  = (l_sh + r_sh) / 2
        mid_hip = (l_hip + r_hip) / 2

        trunk_vec  = mid_sh - mid_hip
        vertical   = np.array([0, -1])
        cos_trunk  = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)
        back_angle = float(np.degrees(np.arccos(np.clip(cos_trunk, -1.0, 1.0))))

        l_elbow_ang = self._angle_3pts(l_sh, l_el, l_wr)
        r_elbow_ang = self._angle_3pts(r_sh, r_el, r_wr)
        elbow_angle = min(l_elbow_ang, r_elbow_ang)

        l_knee_ang  = self._angle_3pts(l_hip, l_kn, l_an)
        r_knee_ang  = self._angle_3pts(r_hip, r_kn, r_an)
        knee_angle  = min(l_knee_ang, r_knee_ang)

        shoulder_diff = abs(float(l_sh[1] - r_sh[1]))
        hip_diff      = abs(float(l_hip[1] - r_hip[1]))
        head_fwd      = abs(float(nose[0] - mid_sh[0]))

        lateral_vec  = mid_sh - mid_hip
        lateral_lean = float(np.degrees(np.arctan2(
            abs(lateral_vec[0]), abs(lateral_vec[1]) + 1e-8
        )))

        return {
            "back_angle":     back_angle,
            "elbow_angle":    elbow_angle,
            "l_elbow_angle":  l_elbow_ang,
            "r_elbow_angle":  r_elbow_ang,
            "knee_angle":     knee_angle,
            "l_knee_angle":   l_knee_ang,
            "r_knee_angle":   r_knee_ang,
            "shoulder_diff":  shoulder_diff,
            "hip_diff":       hip_diff,
            "head_fwd":       head_fwd,
            "lateral_lean":   lateral_lean,
        }

    def calculate_balance(self, lm, w: int, h: int) -> Dict[str, float]:
        px = lambda idx: self._lm_to_px(lm, idx, w, h)
        L  = self.LM

        l_sh  = px(L["l_shoulder"]);  r_sh  = px(L["r_shoulder"])
        l_hip = px(L["l_hip"]);       r_hip = px(L["r_hip"])
        l_kn  = px(L["l_knee"]);      r_kn  = px(L["r_knee"])
        l_an  = px(L["l_ankle"]);     r_an  = px(L["r_ankle"])

        com = (
            0.50 * (l_sh + r_sh) / 2 +
            0.30 * (l_hip + r_hip) / 2 +
            0.20 * (l_kn + r_kn) / 2
        )

        mid_feet = (l_an + r_an) / 2
        mid_sh   = (l_sh + r_sh) / 2
        mid_hip  = (l_hip + r_hip) / 2

        lateral_lean = float((com[0] - mid_feet[0]) / w * 100)
        forward_lean = float((mid_sh[0] - mid_hip[0]) / w * 100)
        foot_spread  = float(abs(l_an[0] - r_an[0]) / w * 100)

        return {
            "com_x":        float(com[0]),
            "com_y":        float(com[1]),
            "lateral_lean": lateral_lean,
            "forward_lean": forward_lean,
            "foot_spread":  foot_spread,
        }
