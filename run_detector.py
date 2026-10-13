import cv2
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
MODEL_PATH = 'id_detector.pt'  # Path to your custom-trained model
TEST_IMAGE_PATH = 'test_images/image1.jpg' # Path to a single test image
OUTPUT_DIR = 'output' # Directory to save results

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD THE MODEL ---
try:
    # Load your custom-trained YOLOv8 model
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'id_detector.pt' is in the same folder as this script.")
    exit()

# Get the class names from the model
class_names = model.names
print(f"Model classes: {class_names}")


# --- CHOOSE YOUR DETECTION MODE ---
# Uncomment one of the following modes to run

# === MODE 1: DETECT ON A SINGLE IMAGE ===
def detect_on_image(image_path):
    """
    Runs detection on a single image and saves the result.
    """
    print(f"\n--- Running detection on single image: {image_path} ---")
    
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Run detection
    # 'conf=0.25' means it will only show detections with > 25% confidence
    results = model(img, conf=0.25)

    # The result is a list, but we use the first element for a single image
    result = results[0]

    # Plot the results on the image
    # This creates a new image with bounding boxes, labels, and confidence scores
    annotated_image = result.plot()

    # --- NEW: ACCESS BOUNDING BOX DATA ---
    print("\n--- Detection Data ---")
    for box in result.boxes:
        # Get coordinates
        xyxy = box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        class_name = class_names[class_id]
        
        print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
        print(f"Coordinates: [x1: {xyxy[0]}, y1: {xyxy[1]}, x2: {xyxy[2]}, y2: {xyxy[3]}]")

        # --- NEXT STEP: YOU CAN CROP THE IMAGE HERE ---
        # cropped_id = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        # cv2.imwrite(os.path.join(OUTPUT_DIR, f"cropped_{class_name}.jpg"), cropped_id)
        # print("Saved cropped image.")
    print("------------------------\n")

    # Save the annotated image
    output_filename = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(output_filename, annotated_image)
    print(f"Saved result to {output_filename}")

    # Display the annotated image in a new window
    cv2.imshow('ID Card Detection Result', annotated_image)
    print("Showing result. Press any key to close...")
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()


# === MODE 2: DETECT IN REAL-TIME (WEBCAM) ===
def detect_on_webcam():
    """
    Runs real-time detection using the webcam.
    """
    print("\n--- Running real-time webcam detection ---")
    print("Press 'q' to quit.")
    
    # 'stream=True' is highly recommended for real-time sources
    # 'source=0' usually refers to the default webcam
    results_generator = model(source=0, stream=True, conf=0.25)

    for result in results_generator:
        # Get the original frame from the result
        original_frame = result.orig_img
        
        # 'result.plot()' draws the bounding boxes on the frame
        annotated_frame = result.plot()

        # --- NEW: ACCESS BOUNDING BOX DATA FOR EACH FRAME ---
        # We check if any detections exist in this frame
        if result.boxes:
            for box in result.boxes:
                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = class_names[class_id]

                # Print the data to your console
                # This will be very fast!
                print(f"LIVE: Detected {class_name} at [{xyxy[0]}, {xyxy[1]}, {xyxy[2]}, {xyxy[3]}] with {confidence:.2f} confidence")

                # --- YOUR NEXT PROJECT: ---
                # 1. CROP THE ID:
                #    cropped_id = original_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                # 
                # 2. RUN OCR (Optical Character Recognition) ON THE CROP:
                #    You'll need a new library like pytesseract or easyocr
                #    text = pytesseract.image_to_string(cropped_id)
                #    print(f"Extracted Text: {text}")
                #
                # 3. TRIGGER AN ACTION:
                #    if confidence > 0.8:
                #        print("ID card verified!")
                #        # (Put logic here to save the image, open a door, etc.)
                
        # Display the frame
        cv2.imshow('Real-time ID Card Detection', annotated_frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()


# --- RUN THE PROGRAM ---
if __name__ == "__main__":
    # --- CHOOSE WHICH MODE TO RUN ---
    
    # To run on a single image:
    # 1. Make sure TEST_IMAGE_PATH is set correctly
    # 2. Uncomment the line below:
    # detect_on_image(TEST_IMAGE_PATH)

    # To run on your webcam:
    # 1. Uncomment the line below:
    detect_on_webcam()