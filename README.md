How It Works
A bowler's video is uploaded → MediaPipe AI detects 33 body landmarks → joint angles are calculated → a risk engine flags dangerous movements → results are shown on a dashboard with an annotated video output.

Tech Stack

Python — core language
MediaPipe — AI pose detection (33 body landmarks)
OpenCV — video processing & drawing overlays
Streamlit — web app interface
NumPy / Pandas — maths & data


Key Features

🦴 Colour-coded skeleton overlay (green → amber → red based on strain)
📐 Live joint angle calculation (back, elbow, knee)
🌡️ Strain heatmap glowing on high-stress joints
⚖️ Centre of mass & balance tracking
📊 Risk dashboard with per-body-zone breakdown
📈 Angle timeline charts across the full bowling action
💡 Personalised injury prevention recommendations
⬇️ Download fully annotated MP4 video
