from typing import Dict, List, Any


# ── Thresholds by sensitivity ─────────────────────────────────────────────────

SENSITIVITY_MULT = {"Low": 1.3, "Medium": 1.0, "High": 0.8}

BASE_THRESHOLDS = {
    # (medium_low, medium_high) → above medium_high = HIGH risk
    "back_angle":    (20, 35),
    "elbow_angle":   (150, 165),    # near-straight = possible chucking + strain
    "knee_angle":    (140, 160),    # near-straight front knee
    "shoulder_diff": (30, 55),      # pixels – asymmetry
    "lateral_lean":  (5, 10),       # % of frame width
    "forward_lean":  (4, 8),
}


class RiskEngine:
    def __init__(self, sensitivity: str = "Medium", bowling_type: str = "Fast"):
        mult = SENSITIVITY_MULT.get(sensitivity, 1.0)
        self.thr = {k: (v[0] * mult, v[1] * mult) for k, v in BASE_THRESHOLDS.items()}
        self.bowling_type = bowling_type

    # ── single-frame risk ────────────────────────────────────────────────────

    def _classify(self, value: float, key: str) -> str:
        lo, hi = self.thr[key]
        if value >= hi:
            return "HIGH"
        if value >= lo:
            return "MEDIUM"
        return "LOW"

    def evaluate(self, angles: Dict, balance: Dict, weight: float, height: float) -> Dict[str, Any]:
        back_risk     = self._classify(angles.get("back_angle",    0), "back_angle")
        elbow_risk    = self._classify(angles.get("elbow_angle",   0), "elbow_angle")
        knee_risk     = self._classify(angles.get("knee_angle",    0), "knee_angle")
        shoulder_risk = self._classify(angles.get("shoulder_diff", 0), "shoulder_diff")
        lateral_risk  = self._classify(abs(balance.get("lateral_lean", 0)), "lateral_lean")
        forward_risk  = self._classify(abs(balance.get("forward_lean", 0)), "forward_lean")

        trunk_risk = "HIGH" if "HIGH" in (lateral_risk, forward_risk) else (
            "MEDIUM" if "MEDIUM" in (lateral_risk, forward_risk) else "LOW")

        risks = [back_risk, elbow_risk, knee_risk, shoulder_risk, trunk_risk]
        overall = "HIGH" if "HIGH" in risks else ("MEDIUM" if "MEDIUM" in risks else "LOW")

        return {
            "back_risk":     back_risk,
            "elbow_risk":    elbow_risk,
            "knee_risk":     knee_risk,
            "shoulder_risk": shoulder_risk,
            "trunk_risk":    trunk_risk,
            "overall":       overall,
        }

    # ── aggregate across frames ──────────────────────────────────────────────

    def aggregate(self, all_risks: List[Dict]) -> Dict[str, Any]:
        if not all_risks:
            return {}

        def majority(key):
            vals = [r[key] for r in all_risks]
            high = vals.count("HIGH") / len(vals)
            med  = vals.count("MEDIUM") / len(vals)
            if high >= 0.15:
                return "HIGH"
            if med >= 0.3:
                return "MEDIUM"
            return "LOW"

        import statistics

        def safe_stat(lst):
            lst = [x for x in lst if x is not None]
            return statistics.mean(lst) if lst else 0.0

        # Pull angle data from risks — we store angles externally, so
        # we just return the majority risk per zone here.
        back_risk     = majority("back_risk")
        elbow_risk    = majority("elbow_risk")
        knee_risk     = majority("knee_risk")
        shoulder_risk = majority("shoulder_risk")
        trunk_risk    = majority("trunk_risk")

        all_overall = [r["overall"] for r in all_risks]
        high_pct    = all_overall.count("HIGH")  / len(all_overall) * 100
        med_pct     = all_overall.count("MEDIUM") / len(all_overall) * 100

        overall = "HIGH" if high_pct >= 15 else ("MEDIUM" if (high_pct + med_pct) >= 30 else "LOW")

        return {
            "overall":        overall,
            "back_risk":      back_risk,
            "elbow_risk":     elbow_risk,
            "knee_risk":      knee_risk,
            "shoulder_risk":  shoulder_risk,
            "trunk_risk":     trunk_risk,
            "high_risk_pct":  high_pct,
            "med_risk_pct":   med_pct,
            # placeholders filled by app from frame_angles list
            "avg_back_angle":    0.0,
            "avg_elbow_angle":   0.0,
        }

    # ── recommendations ──────────────────────────────────────────────────────

    def get_recommendations(self, agg: Dict, bowling_type: str, weight: float, height: float) -> List[Dict]:
        recs = []

        if agg.get("back_risk") in ("HIGH", "MEDIUM"):
            recs.append({
                "icon":  "🧘",
                "title": "Core Strengthening",
                "body":  "Increase core stability training — planks, dead bugs, and rotational exercises help reduce lumbar hyperextension during delivery stride.",
            })

        if agg.get("elbow_risk") in ("HIGH", "MEDIUM"):
            recs.append({
                "icon":  "💪",
                "title": "Arm Action Review",
                "body":  "Elbow angle near full extension may indicate hyperextension or a non-compliant action. Consult a coach for technique adjustment.",
            })

        if agg.get("knee_risk") in ("HIGH", "MEDIUM"):
            recs.append({
                "icon":  "🦵",
                "title": "Front Knee Stiffness",
                "body":  "A straighter front knee increases ground reaction forces. Work on controlled landing mechanics and quad/hamstring flexibility.",
            })

        if agg.get("shoulder_risk") in ("HIGH", "MEDIUM"):
            recs.append({
                "icon":  "🔄",
                "title": "Shoulder Rotation Drill",
                "body":  "Shoulder asymmetry detected. Incorporate shoulder mobility drills and ensure a balanced pre-delivery position to reduce rotator cuff load.",
            })

        if agg.get("trunk_risk") in ("HIGH", "MEDIUM"):
            recs.append({
                "icon":  "⚖️",
                "title": "Balance & Proprioception",
                "body":  "Lateral or forward imbalance during run-up. Single-leg balance training and rhythm drills can improve stability through delivery.",
            })

        if bowling_type == "Fast" and weight > 90:
            recs.append({
                "icon":  "🏃",
                "title": "Load Management",
                "body":  f"At {weight}kg, fast bowling places significant spinal load. Consider monitoring overs per session and scheduling adequate recovery.",
            })

        # Always include rest rec
        recs.append({
            "icon":  "😴",
            "title": "Recovery Protocol",
            "body":  "Ensure 48h recovery between intense bowling sessions. Ice baths, physio massage, and sleep quality all directly reduce injury risk.",
        })

        return recs[:3]
