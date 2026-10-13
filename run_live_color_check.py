import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import os

# --- CONFIGURATION ---
# Path to your YOLO model
YOLO_MODEL_PATH = 'uniform_detector.pt' 
# Path to the color signature file
SIGNATURE_PATH = 'color_signature.npz'

# --- IMPORTANT: SET YOUR VIDEO SOURCE HERE ---
VIDEO_SOURCE = r"C:\Users\syeda\yolo_project\video\IMG_6686.MOV" 
# ---------------------------------------------

# Tolerance for color matching
TOLERANCE = 2.5 
UNIFORM_CLASS_ID = 0 
K_CLUSTERS = 3 
# ---------------------

# 1. Load YOLO Model
print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# 2. Load the Color Signature
print(f"Loading color signature from {SIGNATURE_PATH}...")
try:
    data = np.load(SIGNATURE_PATH)
    MEAN_LAB = data['mean']
    STD_LAB = data['std']
    print("Signature loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find '{SIGNATURE_PATH}'.")
    exit()

# 3. Calculate Acceptable Ranges
min_lab = MEAN_LAB - (STD_LAB * TOLERANCE)
max_lab = MEAN_LAB + (STD_LAB * TOLERANCE)

print("\n--- Calibration Data ---")
print(f"Target Mean (L,a,b): {np.round(MEAN_LAB, 2)}")
print(f"Allowed Range Min:   {np.round(min_lab, 2)}")
print(f"Allowed Range Max:   {np.round(max_lab, 2)}")
print("------------------------\n")

def check_color_compliance(roi):
    try:
        if roi is None or roi.size == 0:
            return False, "ROI Empty", (0,0,0)

        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        pixels = np.float32(lab_roi.reshape(-1, 3))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        counts = np.bincount(labels.flatten())
        dominant_color = centers[np.argmax(counts)]

        is_l_ok = min_lab[0] <= dominant_color[0] <= max_lab[0]
        is_a_ok = min_lab[1] <= dominant_color[1] <= max_lab[1]
        is_b_ok = min_lab[2] <= dominant_color[2] <= max_lab[2]

        if is_l_ok and is_a_ok and is_b_ok:
            return True, f"PASS", dominant_color
        else:
            return False, f"FAIL", dominant_color

    except Exception as e:
        return False, f"Error: {str(e)}", (0,0,0)

# 4. Start Video Feed
print(f"Opening video source: {VIDEO_SOURCE}")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}.")
    print("Check the file path.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # --- NEW: Resize Logic for Laptop Screen ---
    # Max height 800px fits nicely on most 1080p/1200p laptop screens
    target_height = 800 
    height, width = frame.shape[:2]
    
    # Only resize if the video is actually bigger than our screen
    if height > target_height:
        scale = target_height / height
        new_width = int(width * scale)
        display_frame = cv2.resize(frame, (new_width, target_height))
    else:
        display_frame = frame.copy()
    # -------------------------------------------

    # Run YOLO detection (on the original frame for accuracy)
    results = model(frame, verbose=False, conf=0.4)

    # We need to scale the boxes to match the resized display_frame
    scale_factor = display_frame.shape[0] / height

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == UNIFORM_CLASS_ID:
                # Get coordinates from original frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure coordinates are within frame bounds
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Extract ROI from original high-res frame for analysis
                uniform_roi = frame[y1:y2, x1:x2]
                is_compliant, status_text, dom_color = check_color_compliance(uniform_roi)

                color = (0, 255, 0) if is_compliant else (0, 0, 255)
                
                # --- Scale coordinates for the display frame ---
                dx1 = int(x1 * scale_factor)
                dy1 = int(y1 * scale_factor)
                dx2 = int(x2 * scale_factor)
                dy2 = int(y2 * scale_factor)

                cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
                cv2.putText(display_frame, status_text, (dx1, dy1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Video Uniform Color Check", display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()