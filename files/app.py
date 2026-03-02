import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import time
from utils.pose_analyzer import PoseAnalyzer
from utils.risk_engine import RiskEngine
from utils.visualizer import Visualizer

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bowler Biomechanics Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0a0e1a; }

    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #111827 100%); }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #fb7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e2535, #16202f);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }

    .metric-card:hover { transform: translateY(-2px); }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem;
    }

    .risk-low    { color: #4ade80 !important; }
    .risk-medium { color: #fbbf24 !important; }
    .risk-high   { color: #f87171 !important; }

    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-low    { background: #14532d; color: #4ade80; }
    .badge-medium { background: #451a03; color: #fbbf24; }
    .badge-high   { background: #450a0a; color: #f87171; }

    .alert-box {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border-left: 4px solid;
    }

    .alert-low    { background: #052e16; border-color: #4ade80; color: #bbf7d0; }
    .alert-medium { background: #2d1700; border-color: #fbbf24; color: #fef3c7; }
    .alert-high   { background: #2d0000; border-color: #f87171; color: #fecaca; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        border-bottom: 1px solid #1e2d3d;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stProgress"] > div { background: #1e2535; border-radius: 999px; }
    div[data-testid="stProgress"] > div > div { border-radius: 999px; }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: opacity 0.2s;
        width: 100%;
    }

    .stButton > button:hover { opacity: 0.85; }

    .sidebar-info {
        background: #1e2535;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 1rem;
    }

    .phase-tag {
        background: #1e3a5f;
        color: #93c5fd;
        border-radius: 6px;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .stVideo { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    bowler_height = st.slider("Bowler Height (cm)", 150, 210, 175, 1)
    bowler_weight = st.slider("Bowler Weight (kg)", 50, 120, 75, 1)
    bowling_type  = st.selectbox("Bowling Style", ["Fast", "Medium-Fast", "Medium", "Spin"])

    st.markdown("**Sensitivity**")
    sensitivity = st.radio("Detection Sensitivity", ["Low", "Medium", "High"], index=1, horizontal=True)

    st.markdown("**Overlay Options**")
    show_skeleton   = st.checkbox("Show Skeleton", True)
    show_angles     = st.checkbox("Show Joint Angles", True)
    show_com        = st.checkbox("Show Center of Mass", True)
    show_heatmap    = st.checkbox("Show Strain Heatmap", True)
    frame_skip      = st.slider("Process Every N Frames", 1, 5, 2)

    st.markdown("""
    <div class="sidebar-info">
    🏏 <b>Key Risk Zones Monitored</b><br><br>
    🔴 Lower back hyperextension<br>
    🔴 Bowling elbow hyperextension<br>
    🟡 Front knee collapse<br>
    🟡 Shoulder misalignment<br>
    🟢 Head & trunk posture<br><br>
    <i>Based on cricket biomechanics research & ICC guidelines</i>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🏏 Bowler Biomechanics Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered injury prevention & posture analysis using MediaPipe Pose estimation</div>', unsafe_allow_html=True)

# ─── Upload ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload Bowling Video (MP4 / MOV / AVI)",
    type=["mp4", "mov", "avi"],
    help="For best results: side-on or front-on view, clear visibility of full body"
)

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    video_path = tfile.name

    col_vid, col_meta = st.columns([2, 1])

    with col_vid:
        st.markdown('<div class="section-header">📹 Uploaded Video</div>', unsafe_allow_html=True)
        st.video(video_path)

    with col_meta:
        st.markdown('<div class="section-header">👤 Bowler Profile</div>', unsafe_allow_html=True)

        bmi = bowler_weight / ((bowler_height / 100) ** 2)
        bmi_cat = "Underweight" if bmi < 18.5 else ("Normal" if bmi < 25 else ("Overweight" if bmi < 30 else "Obese"))

        st.markdown(f"""
        <div class="metric-card" style="text-align:left; margin-bottom:0.8rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#94a3b8; font-size:0.85rem;">Height</span>
                <span style="color:#f1f5f9; font-weight:700;">{bowler_height} cm</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.4rem;">
                <span style="color:#94a3b8; font-size:0.85rem;">Weight</span>
                <span style="color:#f1f5f9; font-weight:700;">{bowler_weight} kg</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.4rem;">
                <span style="color:#94a3b8; font-size:0.85rem;">BMI</span>
                <span style="color:#fbbf24; font-weight:700;">{bmi:.1f} ({bmi_cat})</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.4rem;">
                <span style="color:#94a3b8; font-size:0.85rem;">Style</span>
                <span style="color:#93c5fd; font-weight:700;">{bowling_type}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Body load estimate
        g_force_approx = {"Fast": 8.5, "Medium-Fast": 7.2, "Medium": 6.0, "Spin": 4.5}
        load = bowler_weight * g_force_approx[bowling_type]
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div style="color:#64748b; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;">Estimated Peak Spinal Load</div>
            <div style="color:#fb7185; font-size:1.6rem; font-weight:800; margin-top:0.3rem;">{load:.0f} N</div>
            <div style="color:#64748b; font-size:0.75rem;">≈ {load/bowler_weight:.1f}x body weight</div>
            <div style="margin-top:0.5rem;">
            <span class="phase-tag">{bowling_type} Bowler Reference</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Run Biomechanics Analysis"):
        analyzer   = PoseAnalyzer()
        risk_eng   = RiskEngine(sensitivity=sensitivity, bowling_type=bowling_type)
        visualizer = Visualizer(
            show_skeleton=show_skeleton,
            show_angles=show_angles,
            show_com=show_com,
            show_heatmap=show_heatmap
        )

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = os.path.join(tempfile.gettempdir(), "analyzed_output.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out      = cv2.VideoWriter(out_path, fourcc, fps / frame_skip, (w, h))

        # ── Progress UI ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">⚙️ Processing Video...</div>', unsafe_allow_html=True)
        prog_bar   = st.progress(0)
        status_txt = st.empty()

        all_risks       = []
        frame_angles    = []
        frame_balance   = []
        frame_idx       = 0
        processed       = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            processed += 1
            prog = min(frame_idx / max(total_frames, 1), 1.0)
            prog_bar.progress(prog)
            status_txt.markdown(
                f"<span style='color:#64748b;'>Frame {frame_idx}/{total_frames} — "
                f"Detected {processed} pose frames</span>",
                unsafe_allow_html=True
            )

            # Pose detection
            landmarks, results = analyzer.process_frame(frame)

            if landmarks:
                # Calculate joint angles
                angles  = analyzer.calculate_angles(landmarks, w, h)
                balance = analyzer.calculate_balance(landmarks, w, h)
                risks   = risk_eng.evaluate(angles, balance, bowler_weight, bowler_height)

                frame_angles.append(angles)
                frame_balance.append(balance)
                all_risks.append(risks)

                # Draw overlays
                frame = visualizer.draw(frame, landmarks, angles, balance, risks, w, h)
            else:
                cv2.putText(frame, "No pose detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            out.write(frame)

        cap.release()
        out.release()
        prog_bar.progress(1.0)
        status_txt.markdown(
            "<span style='color:#4ade80;'>✅ Analysis complete!</span>",
            unsafe_allow_html=True
        )

        # ── Results ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

        if all_risks:
            # Aggregate
            agg = risk_eng.aggregate(all_risks)

            # Top KPIs
            k1, k2, k3, k4 = st.columns(4)
            overall_color = {"LOW": "#4ade80", "MEDIUM": "#fbbf24", "HIGH": "#f87171"}[agg["overall"]]

            with k1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{overall_color};">{agg['overall']}</div>
                    <div class="metric-label">Overall Risk</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{agg['avg_back_angle']:.0f}°</div>
                    <div class="metric-label">Avg Back Angle</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{agg['avg_elbow_angle']:.0f}°</div>
                    <div class="metric-label">Avg Elbow Angle</div>
                </div>""", unsafe_allow_html=True)
            with k4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{agg['high_risk_pct']:.0f}%</div>
                    <div class="metric-label">High-Risk Frames</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Detailed breakdown
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown('<div class="section-header">🔍 Body Part Risk Assessment</div>', unsafe_allow_html=True)

                parts = {
                    "Lower Back":          (agg["back_risk"],     "Hyperextension at front foot landing"),
                    "Bowling Elbow":       (agg["elbow_risk"],    "Straightening / flexion during delivery"),
                    "Front Knee":          (agg["knee_risk"],     "Collapse / valgus stress at release"),
                    "Shoulder Alignment":  (agg["shoulder_risk"], "Rotation imbalance & overreach"),
                    "Head & Trunk":        (agg["trunk_risk"],    "Lateral tilt & forward flexion"),
                }

                for part, (risk, desc) in parts.items():
                    badge_cls = f"badge-{risk.lower()}"
                    alert_cls = f"alert-{risk.lower()}"
                    icon = "🔴" if risk == "HIGH" else ("🟡" if risk == "MEDIUM" else "🟢")
                    st.markdown(f"""
                    <div class="alert-box {alert_cls}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span>{icon} <b>{part}</b></span>
                            <span class="risk-badge {badge_cls}">{risk}</span>
                        </div>
                        <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.3rem;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="section-header">📈 Angle Timeline</div>', unsafe_allow_html=True)

                if frame_angles:
                    import pandas as pd
                    df = pd.DataFrame(frame_angles).fillna(method="ffill")
                    cols_to_plot = [c for c in ["back_angle", "elbow_angle", "knee_angle", "shoulder_diff"] if c in df.columns]
                    if cols_to_plot:
                        st.line_chart(df[cols_to_plot], height=260, use_container_width=True)

                st.markdown('<div class="section-header" style="margin-top:1rem;">⚖️ Balance Over Time</div>', unsafe_allow_html=True)
                if frame_balance:
                    df_bal = pd.DataFrame(frame_balance).fillna(method="ffill")
                    bal_cols = [c for c in ["lateral_lean", "forward_lean"] if c in df_bal.columns]
                    if bal_cols:
                        st.line_chart(df_bal[bal_cols], height=200, use_container_width=True)

            st.markdown("---")

            # Recommendations
            st.markdown('<div class="section-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
            recs = risk_eng.get_recommendations(agg, bowling_type, bowler_weight, bowler_height)
            r1, r2, r3 = st.columns(3)
            for i, (col, rec) in enumerate(zip([r1, r2, r3], recs[:3])):
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:left; height:100%;">
                        <div style="font-size:1.5rem;">{rec['icon']}</div>
                        <div style="color:#e2e8f0; font-weight:600; margin: 0.4rem 0 0.2rem;">{rec['title']}</div>
                        <div style="color:#64748b; font-size:0.82rem;">{rec['body']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Output video ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🎬 Annotated Output Video</div>', unsafe_allow_html=True)
        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Annotated Video",
                    f,
                    file_name="bowler_analysis.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        else:
            st.info("Output video not available — ensure OpenCV is installed correctly.")

else:
    # Empty state
    st.markdown("""
    <div style="
        background: linear-gradient(145deg, #1e2535, #16202f);
        border: 2px dashed #2d3748;
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        margin-top: 2rem;
    ">
        <div style="font-size: 4rem;">🏏</div>
        <div style="color:#e2e8f0; font-size:1.4rem; font-weight:700; margin: 1rem 0 0.5rem;">
            Upload a bowling video to get started
        </div>
        <div style="color:#475569; font-size:0.95rem; max-width:500px; margin:0 auto;">
            The system will detect body pose landmarks, calculate joint angles, estimate balance,
            and flag potential injury risks throughout the bowling action.
        </div>
        <div style="margin-top: 2rem; display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
            <span style="background:#1e3a5f;color:#93c5fd;border-radius:999px;padding:0.4rem 1rem;font-size:0.82rem;">📐 33 Pose Landmarks</span>
            <span style="background:#1a2e1a;color:#4ade80;border-radius:999px;padding:0.4rem 1rem;font-size:0.82rem;">🦴 Skeleton Overlay</span>
            <span style="background:#2d1700;color:#fbbf24;border-radius:999px;padding:0.4rem 1rem;font-size:0.82rem;">⚠️ Injury Risk Alerts</span>
            <span style="background:#2d0000;color:#f87171;border-radius:999px;padding:0.4rem 1rem;font-size:0.82rem;">🌡️ Strain Heatmap</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
