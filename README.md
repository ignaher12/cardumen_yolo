# Cardumen YOLO Metrics

This project evaluates YOLO model performance on videos and saves metrics to a results file.

## Setup Instructions

### 1. Clone the repository (if not already done):
```bash
git clone <repo-url>
cd cardumen_yolo
```

### 2. Create a virtual environment

#### On **macOS/Linux**:
```bash
python3 -m venv .
source bin/activate
```

#### On **Windows** (Command Prompt):
```bat
python -m venv .
Scripts\activate
```

#### On **Windows** (PowerShell):
```powershell
python -m venv .
.\Scripts\Activate.ps1
```

### 3. Install dependencies
Chequear version de CUDA instalada y ver que version de PyTorch corresponde.
https://pytorch.org/get-started/locally/
https://www.digitalocean.com/community/tutorials/yolov8-for-gpu-accelerate-object-detection#step-by-step-guide-to-configure-yolov8-for-gpu

```bash
pip install -r requirements.txt
```

### 4. Place your video files in the `videos/` directory
Supported formats: .mp4, .avi, .mov, .mkv

### 5. Run the metrics script
```bash
python metricas.py
```

### 6. Results
Results will be saved in the `resultados/` directory with a timestamped filename.

---

**Note:**
- Make sure you have Python 3.8+ installed.
- The YOLO model weights (`yolo11n.pt`) should be present in the project root.
