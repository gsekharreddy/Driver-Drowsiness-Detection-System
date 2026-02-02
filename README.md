# **🚗 SafeDrive: AI-Powered Drowsiness Detection System**

SafeDrive is a real-time driver safety system that uses deep learning (CNNs) and computer vision to detect signs of drowsiness. It analyzes facial features and eye closure rates via a webcam and triggers visual alerts on a responsive web dashboard.

Optimized for NVIDIA CUDA Compatable GPUs using Mixed Precision (AMP) training, but capable of running on CPU (if you enjoy waiting). 

# 📸 Demo & Dashboard

The system features a dual-view dashboard:

Driver View: Live feed with Face/Eye tracking and AI Confidence meters.

Simulation: A car that drifts and performs emergency stops based on your alertness.

(Insert your dashboard screenshots here!)

# ⚡ Key Features

Hybrid Detection Logic: Combines a custom CNN model with OpenCV Haar Cascades for robust detection.

Smart "Pop-Eye" Correction: Uses logic to override the AI if eyes are clearly wide open (avoids false alarms).

GPU Acceleration: Fully optimized for CUDA 12.1 with torch.amp (Mixed Precision) for lightning-fast inference.

Web-Based Dashboard: Uses WebSockets to stream analysis to a modern HTML5/JS interface.

Privacy First: All processing happens locally on your machine. No data is sent to the cloud.

# 🛠️ Tech Stack

AI Core: PyTorch (CNN Architecture), Torchvision

Vision: OpenCV (Face/Eye Detection, CLAHE Contrast Enhancement)

Backend: Python websockets + asyncio

Frontend: HTML5, CSS3, Vanilla JavaScript (Canvas API for car simulation)

# ⚙️ Prerequisites

Hardware: Webcam. NVIDIA GPU recommended (RTX 2050 or better) for high FPS.

OS: Windows 10/11 or Linux.

Python: 3.10 (Strict requirement. Newer versions may break torch-gpu wheels).

# 🚀 Installation

_1. Clone the Repository_
```
git clone https://github.com/gsekharreddy/Driver-Drowsiness-Detection-System
```

_2. Install Conda (Miniconda)_

Don't install the full "Anaconda" (it's 5GB of bloat). Install Miniconda (it's just the engine).

2.1> Download: Go to Miniconda for Windows and grab the Windows 64-bit installer.

2.2> Install: Run it.

2.3> Crucial Step: When asked, check the box that says "Add Miniconda3 to my PATH environment variable" (even if it says it's not recommended). It makes life easier. 

2.4> Verify: Open your terminal (cmd/PowerShell) and type:
```
conda --version
```
_3. The Setup Commands_
**Run these inside your Anaconda Prompt or Terminal.**

**Step A:** Create the Environment (Python 3.10)
Always start fresh. Python 3.10 is the stability king for AI right now.

```Bash
conda create -n safe_drive python=3.10 -y
conda activate safe_drive
```
**Step B:** Install PyTorch (The Heavy Lifter) 🏋️‍♂️
This command pulls from the pytorch and nvidia channels to get the CUDA 12.1 drivers. (Note: This is the one that might take 10 minutes to "solve environment.)

```Bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
Step C: Install The Rest (Vision & Web) 👁️
We use conda-forge (a community channel) because the default channel often has outdated versions of OpenCV.

```Bash
conda install -c conda-forge opencv matplotlib jupyter pillow websockets numpy -y
```
**Summary of installed packages:**

pytorch, torchvision: The Brains (AI/Deep Learning).

pytorch-cuda: The Muscle (Drivers for your RTX GPU).

opencv: The Eyes (Webcam access, face detection).

websockets: The Mouth (Talks to the HTML dashboard).

pillow: Image processing helper.

numpy: Math helper.

jupyter: The Notebook interface.

**Create the environment**
```
conda create -n drowsiness_env python=3.10 -y
```
**Activate it**
```
conda activate drowsiness_env
```

_4. Install Dependencies_

For GPU Users (NVIDIA):
Run this exact command to get the CUDA 12.1 drivers:
```
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

For Everyone:
Install the rest of the project requirements:
```
pip install websockets opencv-python pillow numpy matplotlib jupyter
```

# 🎮 Usage

Step 1: Train the Model (Optional)/Use the 'drowsiness_model_gpu.pth' model.

If you want to train the model yourself (and verify your GPU works):

**Train.ipynb has the code to train the Model**


Note: This script automatically handles .ipynb_checkpoints cleanup.

Step 2: Start the AI Server

This script loads the model (.pth file) and starts the WebSocket server.

python drive_server_gpu.py


Wait until you see: 🚀 AI Server started on ws://localhost:8765

Step 3: Launch the Dashboard

Open dashboard_video.html in your web browser (Chrome/Edge/Firefox).

Allow Camera Permissions.

Drive safely! (Or close your eyes to watch the car crash). 🚗💥

# 📂 Project Structure

safedrive-ai/

├── data/                    # Dataset (Train/Val/Test)

├── drowsiness_model_gpu.pth # Trained Model Weights

├── drive_server_gpu.py     # Python Backend (WebSockets + Inference)

├── Train.ipynb   # Training Script

├── dashboard.html    # Frontend Interface

├── Verification.ipynb #Test if your GPU is compatible for Accelerated Hardware Training

└── README.md               # You are here


# 🤝 Contributing

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

# 📜 License

Distributed under the MIT License. See LICENSE for more information.

Note: If you encounter RuntimeError: Torch not compiled with CUDA enabled, please verify you installed the specific CUDA 12.1 wheel listed in step 3. Do not trust pip install torch blindly. 
