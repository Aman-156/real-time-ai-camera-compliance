import cv2
import numpy as np
import glob
import os
from sklearn.cluster import KMeans

# --- UPDATE THIS PATH CAREFULLY ---
# Make sure this path leads to the folder that CONTAINS the 'images' and 'labels' folders.
DATASET_DIR = r"C:\Users\syeda\yolo_project\uniform--detection-5" 
# ----------------------------------

IMAGE_DIR = r"C:\Users\syeda\yolo_project\uniform--detection-5\train\images"
LABEL_DIR = r"C:\Users\syeda\yolo_project\uniform--detection-5\train\labels"
K_CLUSTERS = 3 

print(f"--- DIAGNOSTIC CHECK ---")
print(f"Checking Main Directory: {DATASET_DIR}")

if not os.path.exists(DATASET_DIR):
    print(f"❌ ERROR: The directory '{DATASET_DIR}' does not exist.")
    print("   Please check if you unzipped the file and copied the path correctly.")
    exit()

if not os.path.exists(IMAGE_DIR):
    print(f"❌ ERROR: Could not find an 'images' folder inside the path.")
    print(f"   The script expects to find: {IMAGE_DIR}")
    print(f"   Current contents of folder: {os.listdir(DATASET_DIR)}")
    exit()

print(f"✅ 'images' folder found.")

# Check for multiple file extensions
image_files = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

print(f"Found {len(image_files)} images.")

if len(image_files) == 0:
    print("❌ ERROR: 'images' folder exists, but it is empty or contains no image files.")
    exit()

# --- START PROCESSING ---
all_dominant_colors = []
print("Processing images...")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape

    # Handle different extension mismatches between image and label
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABEL_DIR, base_name + ".txt")

    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        try:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                class_id = int(parts[0]) 
                
                # Only process if class_id is 0 (Adjust if your uniform is a different class)
                if class_id == 0: 
                    x_c, y_c, w_norm, h_norm = map(float, parts[1:5])
                    x_min = int((x_c - w_norm/2) * w)
                    y_min = int((y_c - h_norm/2) * h)
                    x_max = int((x_c + w_norm/2) * w)
                    y_max = int((y_c + h_norm/2) * h)

                    # Clamp coordinates to image boundaries
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)

                    roi = img[y_min:y_max, x_min:x_max]
                    if roi.size == 0:
                        continue

                    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                    pixels = np.float32(lab_roi.reshape(-1, 3))
                    
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(pixels, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    
                    counts = np.bincount(labels.flatten())
                    dominant_color = centers[np.argmax(counts)]
                    all_dominant_colors.append(dominant_color)
                    # Break after finding the first uniform in the image (to speed up)
                    break 

        except Exception as e:
            print(f"Skipping file due to error: {e}")

print(f"Successfully processed {len(all_dominant_colors)} samples.")

if all_dominant_colors:
    colors_array = np.array(all_dominant_colors)
    mean_lab = np.mean(colors_array, axis=0)
    std_lab = np.std(colors_array, axis=0)

    print("\n--- CALIBRATION RESULTS ---")
    print(f"Mean: {mean_lab}")
    print(f"Std Dev: {std_lab}")
    
    np.savez('color_signature.npz', mean=mean_lab, std=std_lab)
    print("✅ Saved 'color_signature.npz'. You are ready for the live script.")
else:
    print("⚠️ Processed images but found no valid labels for Class 0.")
    print("Check if your uniform is actually Class 0 in data.yaml")