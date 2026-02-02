# -----------------------------------------------------------------------------------------
# 📦 DEPENDENCY SETUP:
# pip install websockets opencv-python pillow numpy
# -----------------------------------------------------------------------------------------

import asyncio
import websockets
import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import base64

# --- 1. MODEL DEFINITION ---
class DrowsinessNet(nn.Module):
    def __init__(self):
        super(DrowsinessNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. LOAD MODEL ON GPU ---
print("Loading Model on RTX 2050...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrowsinessNet().to(device)

try:
    model.load_state_dict(torch.load("drowsiness_model_gpu.pth", map_location=device))
    model.eval()
    print(f"✅ Model Loaded on {device}!")
except FileNotFoundError:
    print("❌ Model file not found! Make sure 'drowsiness_model_gpu.pth' is here.")
    exit()
except RuntimeError as e:
    print(f"❌ Architecture Mismatch: {e}")
    exit()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- 3. DETECTION SETUP ---
# Load Face & Eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Contrast Enhancer (CLAHE) - Helps with bad lighting
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# --- 4. SERVER LOGIC ---
async def send_status(websocket):
    print("Client connected! Starting webcam...")
    cap = cv2.VideoCapture(0)
    consecutive_sleepy_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Improve Lighting
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_enhanced = clahe.apply(gray) # Apply contrast boost

            # 2. Face Detection
            faces = face_cascade.detectMultiScale(gray_enhanced, 1.1, 4)

            status = "NORMAL"
            score = 0.0
            
            if len(faces) > 0:
                # Get largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                
                # --- HYBRID LOGIC START ---
                # A. Look for Eyes inside the face
                roi_gray = gray_enhanced[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
                
                # Draw Box for Face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw Boxes for Eyes (Visual feedback)
                for (ex, ey, ew, eh) in eyes:
                    # Draw on the main frame (offset by face position)
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 1)

                # B. Crop Face for AI (Tighter Crop to remove background)
                # Zoom in 10% to focus on features
                crop_x = x + int(w*0.1)
                crop_y = y + int(h*0.1)
                crop_w = int(w*0.8)
                crop_h = int(h*0.8)
                
                # Safety check for bounds
                if crop_w > 10 and crop_h > 10:
                    face_crop = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    
                    # Convert to Tensor
                    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_face)
                    input_tensor = transform(pil_image).unsqueeze(0).to(device)
                    
                    # AI Predict
                    with torch.no_grad():
                        logits = model(input_tensor)
                        score = torch.sigmoid(logits).item() 
                
                # C. The "Pop Eye" Correction
                # If AI says Sleepy (>0.5) BUT we clearly see 2 open eyes, FORCE score down.
                if len(eyes) >= 2:
                    # 'Hybrid Weighting': Trust eyes more than AI for "Awake"
                    score = score * 0.4 
                    cv2.putText(frame, "EYES OPEN (Override)", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                
                # Logic Thresholds (Stricter)
                if score > 0.75: # Was 0.5, increased to reduce false alarms
                    consecutive_sleepy_frames += 1
                else:
                    # Recovery is faster now
                    consecutive_sleepy_frames = max(0, consecutive_sleepy_frames - 2)
                    
                if consecutive_sleepy_frames > 6: status = "DROWSY"
                if consecutive_sleepy_frames > 20: status = "SLEEP"
                
                # Draw Score
                label = f"Sleep Score: {score:.2f}"
                color = (0, 0, 255) if score > 0.75 else (0, 255, 0)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            else:
                cv2.putText(frame, "NO FACE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                consecutive_sleepy_frames = 0

            # 3. Send to Browser
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            data = {
                "status": status,
                "score": score,
                "image": jpg_as_text
            }
            
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.04) 

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        cap.release()

async def main():
    print("🚀 AI Server started on ws://localhost:8765")
    async with websockets.serve(send_status, "localhost", 8765):
        await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping server.")