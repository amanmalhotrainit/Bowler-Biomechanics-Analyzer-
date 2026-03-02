import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, Any


# Risk colour map
RISK_COLOUR = {
    "LOW":    (74,  222, 128),   # green
    "MEDIUM": (251, 191,  36),   # amber
    "HIGH":   (248, 113, 113),   # red
}

# Skeleton connections (MediaPipe 33-point body)
SKELETON_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27), (27, 31), (27, 29),
    # Right leg
    (24, 26), (26, 28), (28, 32), (28, 30),
]

# Which landmark indices belong to which risk zone
ZONE_LANDMARKS = {
    "back_risk":     [11, 12, 23, 24],
    "elbow_risk":    [13, 14, 15, 16],
    "knee_risk":     [25, 26, 27, 28],
    "shoulder_risk": [11, 12],
    "trunk_risk":    [11, 12, 23, 24, 0],
}


class Visualizer:
    def __init__(
        self,
        show_skeleton: bool = True,
        show_angles:   bool = True,
        show_com:      bool = True,
        show_heatmap:  bool = True,
    ):
        self.show_skeleton = show_skeleton
        self.show_angles   = show_angles
        self.show_com      = show_com
        self.show_heatmap  = show_heatmap

    # ── main draw ────────────────────────────────────────────────────────────

    def draw(
        self,
        frame:   np.ndarray,
        lm,
        angles:  Dict[str, float],
        balance: Dict[str, float],
        risks:   Dict[str, str],
        w: int,
        h: int,
    ) -> np.ndarray:

        out = frame.copy()

        px = lambda idx: (int(lm[idx].x * w), int(lm[idx].y * h))

        if self.show_heatmap:
            out = self._draw_heatmap(out, lm, risks, w, h, px)

        if self.show_skeleton:
            out = self._draw_skeleton(out, lm, risks, w, h, px)
            out = self._draw_joints(out, lm, risks, w, h, px)

        if self.show_angles:
            out = self._draw_angles(out, lm, angles, risks, w, h, px)

        if self.show_com:
            out = self._draw_com(out, balance, risks)

        out = self._draw_hud(out, risks, angles, balance, w, h)

        return out

    # ── heatmap glow ─────────────────────────────────────────────────────────

    def _draw_heatmap(self, frame, lm, risks, w, h, px):
        overlay = frame.copy()
        for zone, lm_ids in ZONE_LANDMARKS.items():
            risk  = risks.get(zone, "LOW")
            if risk == "LOW":
                continue
            colour = RISK_COLOUR[risk]
            alpha  = 0.18 if risk == "MEDIUM" else 0.30
            for idx in lm_ids:
                cx, cy = px(idx)
                radius = 45 if risk == "HIGH" else 35
                cv2.circle(overlay, (cx, cy), radius, colour, -1)
        return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # ── skeleton lines ────────────────────────────────────────────────────────

    def _draw_skeleton(self, frame, lm, risks, w, h, px):
        # Build a per-landmark risk colour
        lm_risk = ["LOW"] * 33
        for zone, lm_ids in ZONE_LANDMARKS.items():
            zone_risk = risks.get(zone, "LOW")
            for idx in lm_ids:
                # escalate
                current = lm_risk[idx]
                if zone_risk == "HIGH" or (zone_risk == "MEDIUM" and current == "LOW"):
                    lm_risk[idx] = zone_risk

        for (a, b) in SKELETON_CONNECTIONS:
            if a >= len(lm) or b >= len(lm):
                continue
            risk_a  = lm_risk[a]
            risk_b  = lm_risk[b]
            # use higher risk for the segment colour
            seg_risk = "HIGH" if "HIGH" in (risk_a, risk_b) else (
                        "MEDIUM" if "MEDIUM" in (risk_a, risk_b) else "LOW")
            colour  = RISK_COLOUR[seg_risk]
            thick   = 3 if seg_risk == "HIGH" else 2
            cv2.line(frame, px(a), px(b), colour, thick, cv2.LINE_AA)

        return frame

    # ── joint dots ────────────────────────────────────────────────────────────

    def _draw_joints(self, frame, lm, risks, w, h, px):
        key_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        lm_risk    = self._landmark_risks(risks)

        for idx in key_joints:
            if idx >= len(lm):
                continue
            risk   = lm_risk[idx]
            colour = RISK_COLOUR[risk]
            cx, cy = px(idx)
            r = 7 if risk == "HIGH" else 5
            cv2.circle(frame, (cx, cy), r + 3, (20, 20, 20), -1)
            cv2.circle(frame, (cx, cy), r,     colour,       -1)

        return frame

    def _landmark_risks(self, risks):
        lm_risk = ["LOW"] * 33
        for zone, lm_ids in ZONE_LANDMARKS.items():
            zone_risk = risks.get(zone, "LOW")
            for idx in lm_ids:
                current = lm_risk[idx]
                if zone_risk == "HIGH" or (zone_risk == "MEDIUM" and current == "LOW"):
                    lm_risk[idx] = zone_risk
        return lm_risk

    # ── angle labels ─────────────────────────────────────────────────────────

    def _draw_angles(self, frame, lm, angles, risks, w, h, px):
        labels = [
            (23, f"Back {angles.get('back_angle',0):.0f}°",   risks.get("back_risk",  "LOW")),
            (13, f"Elbow {angles.get('l_elbow_angle',0):.0f}°",risks.get("elbow_risk","LOW")),
            (14, f"Elbow {angles.get('r_elbow_angle',0):.0f}°",risks.get("elbow_risk","LOW")),
            (25, f"Knee {angles.get('l_knee_angle',0):.0f}°",  risks.get("knee_risk", "LOW")),
            (26, f"Knee {angles.get('r_knee_angle',0):.0f}°",  risks.get("knee_risk", "LOW")),
        ]
        for idx, text, risk in labels:
            if idx >= len(lm):
                continue
            cx, cy = px(idx)
            colour = RISK_COLOUR[risk]
            # shadow
            cv2.putText(frame, text, (cx + 12, cy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (cx + 12, cy),     cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour,    1, cv2.LINE_AA)
        return frame

    # ── centre of mass indicator ──────────────────────────────────────────────

    def _draw_com(self, frame, balance, risks):
        cx = int(balance.get("com_x", 0))
        cy = int(balance.get("com_y", 0))
        overall = risks.get("overall", "LOW")
        colour  = RISK_COLOUR[overall]

        # outer ring
        cv2.circle(frame, (cx, cy), 14, (20, 20, 20), -1)
        cv2.circle(frame, (cx, cy), 12, colour,        2)
        # crosshair
        cv2.line(frame, (cx - 8, cy), (cx + 8, cy), colour, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 8), (cx, cy + 8), colour, 1, cv2.LINE_AA)
        # label
        cv2.putText(frame, "COM", (cx + 16, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        return frame

    # ── HUD overlay ───────────────────────────────────────────────────────────

    def _draw_hud(self, frame, risks, angles, balance, w, h):
        # semi-transparent top bar
        bar_h   = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 20, 30), -1)
        frame   = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        overall = risks.get("overall", "LOW")
        colour  = RISK_COLOUR[overall]
        label   = f"RISK: {overall}"
        cv2.putText(frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

        # mini stats on right
        stats = [
            f"Back {angles.get('back_angle',0):.0f}°",
            f"Elbow {angles.get('elbow_angle',0):.0f}°",
            f"Knee {angles.get('knee_angle',0):.0f}°",
        ]
        x_offset = w - 260
        for i, s in enumerate(stats):
            cv2.putText(frame, s, (x_offset + i * 85, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        return frame
