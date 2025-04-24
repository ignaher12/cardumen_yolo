# Cardumen YOLO Metrics

This project evaluates YOLO model performance on videos and saves metrics to a results file.

## Setup Instructions

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repo-url>
   cd cardumen_yolo
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your video files** in the `videos/` directory (supported: .mp4, .avi, .mov, .mkv).

5. **Run the metrics script:**
   ```bash
   python metricas.py
   ```

6. **Results** will be saved in the `resultados/` directory with a timestamped filename.

---

**Note:**
- Make sure you have Python 3.8+ installed.
- The YOLO model weights (`yolo11n.pt`) should be present in the project root.
