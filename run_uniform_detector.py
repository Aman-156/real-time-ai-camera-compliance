import cv2
from roboflow import Roboflow
import os

# --- CONFIGURATION ---
# 1. Get your API key from Roboflow:
#    (Account Settings -> Roboflow API -> Private API Key)
ROBOFLOW_API_KEY = "TaSzYlMync2Nrwb7od6N"

# 2. Get your model URL from Roboflow:
#    (Dataset -> Version -> Deploy -> Roboflow Hosted API)
#    It looks like "project-name/version-number"
MODEL_ID = "uniform detection\ 2"

# 3. Set your test image path
TEST_IMAGE_PATH = "test_images/image1.jpg"
# ---------------------

# --- VALIDATION ---
if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE" or MODEL_ID == "your-project-name/1":
    print("Error: Please update 'ROBOFLOW_API_KEY' and 'MODEL_ID' at the top of this script.")
    print("You can get these from your Roboflow project's 'Deploy' tab.")
    exit()

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
    print("Please make sure you have a 'test_images' folder with 'image1.jpg' in it.")
    exit()

print("Connecting to Roboflow...")
# Initialize the Roboflow object
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(MODEL_ID.split('/')[0])
    model = project.version(int(MODEL_ID.split('/')[1])).model
    print("Successfully connected to Roboflow.")
except Exception as e:
    print(f"Error connecting to Roboflow. Check your API key and Model ID. {e}")
    exit()


# --- RUN INFERENCE ---
print(f"\nRunning inference on '{TEST_IMAGE_PATH}' via Roboflow Cloud API...")
try:
    # Run prediction on the image
    prediction = model.predict(TEST_IMAGE_PATH, confidence=40, overlap=30).json()
    print("Prediction successful.")
    
    # --- PROCESS RESULTS ---
    print("\n--- Detection Data ---")
    if not prediction['predictions']:
        print("No objects detected.")
    else:
        # Load the image to draw on
        img = cv2.imread(TEST_IMAGE_PATH)
        
        for pred in prediction['predictions']:
            class_name = pred['class']
            conf = pred['confidence']
            
            # Get coordinates
            x1 = int(pred['x'] - pred['width'] / 2)
            y1 = int(pred['y'] - pred['height'] / 2)
            x2 = int(pred['x'] + pred['width'] / 2)
            y2 = int(pred['y'] + pred['height'] / 2)
            
            print(f"Detected: {class_name} with {conf*100:.1f}% confidence.")

            # Draw the box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the label
            label = f"{class_name} {conf*100:.1f}%"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save and show the image
        output_filename = "output/api_test_result.jpg"
        cv2.imwrite(output_filename, img)
        print("------------------------\n")
        print(f"Saved result to {output_filename}")

        cv2.imshow("Roboflow API Result", img)
        print("Showing result. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")
    print("This can happen if your model isn't deployed or your API key is wrong.")