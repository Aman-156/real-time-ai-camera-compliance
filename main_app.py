import cv2
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
import os
from plyer import notification
import sys
import time
import math # Import math for checking finite numbers
import pywhatkit # --- NEW: Import for WhatsApp ---
# --- REMOVED: deque is no longer needed ---
# from collections import deque 
from deepface import DeepFace # --- RE-ADDED: For advanced facial recognition ---
import tkinter as tk # --- NEW: For the source selection UI ---
from tkinter import filedialog, ttk # --- NEW: For UI elements ---
import pandas as pd # --- NEW: For Excel Report ---
from sklearn.cluster import KMeans # --- NEW: For Color Check ---

# --- REMOVED: Old source configuration is gone ---
VIDEO_PLAYBACK_SPEED_FACTOR = 1.0
# ---------------------------------------------

# --- NEW: WHATSAPP PHONE BOOK ---
PHONE_BOOK = {
    "numbers
    "unknown": None 
}
# ----------------------------------

# --- NEW: Face Database Path ---
FACE_DATABASE_PATH = os.path.abspath("face_database")
# -------------------------------

# --- CONFIGURATION PARAMETERS ---
FRAME_SKIP = 15 # We will process 1 in every 15 frames.
# ------------------------------

# --- MODIFIED: Back to "Grace Period" for faster testing ---
VIOLATION_GRACE_PERIOD = 3 # Set low for fast violation triggers
# ----------------------------------------------------
# --- NEW: Tracker Time-to-Live ---
CLEANUP_TTL = 50 # 50 frames
# ---------------------------------

# --- NEW: Compliance Confidence Threshold ---
# If the AI's confidence is LESS than this (e.g., 50%), it counts as a violation.
COMPLIANCE_CONFIDENCE_THRESHOLD = 50.0 
# ------------------------------------------

# --- NEW: HARDCODED COLOR CALIBRATION DATA ---
# Based on your Phase A calibration results
CALIB_MEAN_LAB = np.array([125.89, 126.78, 119.4]) 
CALIB_MIN_LAB = np.array([-22.94, 121.68, 103.54])
CALIB_MAX_LAB = np.array([274.72, 131.89, 135.27])
K_CLUSTERS = 3
# ---------------------------------------------


if 'pytest' not in sys.modules: # Skip UI when testing
    # --- NEW: Source Chooser Class (Tkinter UI) ---
    class SourceChooser:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Select Video Source")
            self.root.geometry("400x200")
            
            # --- NEW: Get Screen Dimensions for Auto-Sizing ---
            self.screen_width = self.root.winfo_screenwidth()
            self.screen_height = self.root.winfo_screenheight()
            # --------------------------------------------------
            
            self.mode = None
            self.sources = []

            # Style
            style = ttk.Style(self.root)
            style.configure('TButton', font=('Helvetica', 12), padding=10)
            
            main_frame = ttk.Frame(self.root, padding="20 20 20 20")
            main_frame.pack(expand=True, fill='both')
            
            label = ttk.Label(main_frame, text="Choose your video source:", font=('Helvetica', 14))
            label.pack(pady=(0, 20))

            # Button for Webcam 0 (Built-in)
            btn_webcam0 = ttk.Button(main_frame, text="Live (Built-in Webcam)", command=lambda: self.on_webcam(0))
            btn_webcam0.pack(fill='x', pady=5)
            
            # Button for Webcam 1 (External/USB)
            btn_webcam1 = ttk.Button(main_frame, text="Live (External USB Cam)", command=lambda: self.on_webcam(1))
            btn_webcam1.pack(fill='x', pady=5)

            # Button for File Upload
            btn_upload = ttk.Button(main_frame, text="Upload Video Files", command=self.on_upload)
            btn_upload.pack(fill='x', pady=5)
            
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        def on_webcam(self, index):
            print(f"Mode selected: Webcam (Index: {index})")
            self.mode = 'webcam'
            self.sources = [index]
            self.root.destroy()

        def on_upload(self):
            print("Mode selected: Upload")
            # Open file dialog to select multiple video files
            files = filedialog.askopenfilenames(
                title="Select Video Files",
                filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
            )
            if files:
                self.mode = 'video'
                self.sources = list(files)
                print(f"Selected {len(self.sources)} file(s): {self.sources}")
                self.root.destroy()
            else:
                print("No files selected.")
                
        def on_close(self):
            print("No source selected. Exiting.")
            self.root.destroy()

        def run(self):
            self.root.mainloop()
            # Return screen dimensions along with choices
            return self.mode, self.sources, self.screen_width, self.screen_height
    # --- End of Source Chooser Class ---

# --- NEW: Helper function to send WhatsApp Alert ---
def send_whatsapp_violation(name, violation_type, source_name, snapshot_path):
    """
    Looks up a person's phone number and sends them a WhatsApp alert
    with the violation snapshot.
    """
    # --- FIXED: Convert name to lowercase for lookup ---
    lookup_name = name.lower()
    phone_number = PHONE_BOOK.get(lookup_name)
    # -------------------------------------------------
    
    if phone_number:
        message = f"--- AI COMPLIANCE ALERT ---\n"
        message += f"Source: {source_name}\n"
        message += f"Person: {name}\n" # Display original name in message
        message += f"Violation: {violation_type}\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            print(f"Sending WhatsApp alert to {name} ({phone_number}) for {violation_type}...")
            # This will open WhatsApp Web, send the image with the caption, and close the tab
            pywhatkit.sendwhats_image(
                phone_number, 
                snapshot_path, 
                message, 
                wait_time=15, # Time to wait for web.whatsapp to load
                tab_close=True
            )
            print(f"Successfully sent alert to {name}.")
        except Exception as e:
            print(f"Error sending WhatsApp message to {name}: {e}")
            print("NOTE: The first time you run this, you must scan the WhatsApp Web QR code.")
    else:
        # Fixed print message to be more helpful
        print(f"Cannot send WhatsApp: No phone number found for '{name}' (looked up as '{lookup_name}').")


# --- NEW: Function to load violations from CSV at startup ---
def load_todays_violations(csv_path):
    """
    Reads the 'violations_today.csv' file to prevent duplicate logging.
    """
    todays_violations = set()
    today_date = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(csv_path):
        # CSV doesn't exist yet, so no violations to load.
        return todays_violations

    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip the header row
            
            name_idx = 3
            violation_idx = 4
            timestamp_idx = 0 

            for row in reader:
                try:
                    timestamp_str = row[timestamp_idx]
                    violation_date = datetime.fromisoformat(timestamp_str).strftime("%Y-%m-%d")

                    if violation_date == today_date:
                        person_name = row[name_idx].lower()
                        violation_type = row[violation_idx]
                        
                        violation_key = (person_name, violation_type)
                        todays_violations.add(violation_key)
                        
                except (IndexError, ValueError):
                     print(f"Warning: Skipping malformed row in {csv_path}: {row}")
    
    except Exception as e:
        print(f"Error loading violations log {csv_path}: {e}. Starting with an empty log.")
        return set() # Return an empty set on error
    
    if todays_violations:
        print(f"Successfully loaded {len(todays_violations)} violations from today. These will not be logged again.")
    else:
        print("No previous violations found for today.")
        
    return todays_violations
# --- End of new function ---

# --- NEW: Function to clear *daily* violations folder ---
def clear_violations_for_new_day(violations_dir_path, daily_csv_path):
    """
    Checks if it's a new day. If it is, it clears the *daily* log
    and all old snapshots. The *master log* is never touched.
    """
    date_tracker_file = os.path.join(violations_dir_path, 'last_run_date.txt')
    today_date = datetime.now().strftime("%Y-%m-%d")
    last_run_date = ""

    # 1. Read the last run date from our tracker file
    try:
        if os.path.exists(date_tracker_file):
            with open(date_tracker_file, 'r') as f:
                last_run_date = f.read().strip()
    except Exception as e:
        print(f"Warning: Could not read date tracker file: {e}")

    # 2. Compare dates
    if today_date != last_run_date:
        print(f"--- New Day Detected ({today_date}) ---")
        print("Clearing all *old snapshots* and the *daily log*...")
        try:
            # 3a. Delete all files in the directory *except* the master log
            for filename in os.listdir(violations_dir_path):
                file_path = os.path.join(violations_dir_path, filename)
                
                if filename == 'violations_master_log.csv' or filename == 'last_run_date.txt':
                    continue # DO NOT DELETE THESE
                    
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            print("Old snapshots and daily log cleared.")

            # 3b. Re-create the DAILY CSV header
            with open(daily_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(['Timestamp', 'SourceIdx', 'ID', 'Name', 'ViolationType', 'SnapshotPath'])
            
            # 3c. Write the new date to the tracker file
            with open(date_tracker_file, 'w') as f:
                f.write(today_date)
                
        except Exception as e:
            print(f"Error while clearing violations: {e}")
    else:
        # It's the same day, just a script restart
        print(f"Resuming logging for {today_date}.")
# --- End of new function ---

# --- NEW: Monthly Report Generator ---
def generate_monthly_report(master_log_path, violations_dir):
    """
    Reads the master log and generates a monthly Excel report
    for the *current* month.
    """
    print("Generating this month's report...")
    
    current_month = datetime.now().month
    current_year = datetime.now().year
    report_filename = os.path.join(violations_dir, f"Monthly_Report_{current_year}_{current_month:02d}.xlsx")

    try:
        if not os.path.exists(master_log_path):
            print("No 'violations_master_log.csv' found. Skipping report.")
            return

        # Read the master log
        df = pd.read_csv(master_log_path)
        if df.empty:
            print("Master log is empty. Skipping report.")
            return
            
        # Convert timestamp column to datetime objects
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Filter for the current month and year
        df_month = df[
            (df['Timestamp'].dt.month == current_month) &
            (df['Timestamp'].dt.year == current_year)
        ].copy()

        if df_month.empty:
            print(f"No violations found for {current_year}-{current_month:02d}. No report generated.")
            return

        # --- Create Analytics ---
        # 1. Pivot Table: Violations by Person and Type
        summary_pivot = pd.pivot_table(
            df_month,
            index='Name',
            columns='ViolationType',
            values='Timestamp',
            aggfunc='count',
            fill_value=0,
            margins=True, # Adds "All" (total) row/column
            margins_name='Total'
        )
        
        # 2. Top Violators
        top_violators = df_month['Name'].value_counts().reset_index()
        top_violators.columns = ['Person', 'Total Violations']
        
        # 3. Most Common Violations
        common_violations = df_month['ViolationType'].value_counts().reset_index()
        common_violations.columns = ['Violation Type', 'Total Count']

        # --- Save to Excel File ---
        with pd.ExcelWriter(report_filename, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_pivot.to_excel(writer, sheet_name='Summary', startrow=1)
            top_violators.to_excel(writer, sheet_name='Summary', startrow=len(summary_pivot) + 5)
            common_violations.to_excel(writer, sheet_name='Summary', startrow=len(summary_pivot) + len(top_violators) + 10)
            
            # Add titles to summary sheet
            ws = writer.sheets['Summary']
            ws['A1'] = "Violation Summary (by Person & Type)"
            ws[f'A{len(summary_pivot) + 5}'] = "Top Violators"
            ws[f'A{len(summary_pivot) + len(top_violators) + 10}'] = "Most Common Violations"
            
            # Sheet 2: Raw Data for the month
            df_month.to_excel(writer, sheet_name='RawData_ThisMonth', index=False)

        print(f"Successfully generated monthly report: {report_filename}")

    except Exception as e:
        print(f"Error generating monthly report: {e}")
# --- End of Report Generator ---


# --- NEW: Color Check Function ---
def check_color_compliance(roi):
    """
    Analyzes the dominant color of the ROI using K-Means and
    compares it to the HARDCODED calibration signature.
    """
    try:
        if roi is None or roi.size == 0:
            return False

        # Convert ROI to L*a*b* color space
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        pixels = np.float32(lab_roi.reshape(-1, 3))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        counts = np.bincount(labels.flatten())
        dominant_color = centers[np.argmax(counts)]
        
        # --- MODIFIED: Widen the range by +/- 20 to handle different lighting ---
        TOLERANCE_BUFFER = 20
        
        # Check if the dominant color is within the allowed range
        is_l_ok = (CALIB_MIN_LAB[0] - TOLERANCE_BUFFER) <= dominant_color[0] <= (CALIB_MAX_LAB[0] + TOLERANCE_BUFFER)
        is_a_ok = (CALIB_MIN_LAB[1] - TOLERANCE_BUFFER) <= dominant_color[1] <= (CALIB_MAX_LAB[1] + TOLERANCE_BUFFER)
        is_b_ok = (CALIB_MIN_LAB[2] - TOLERANCE_BUFFER) <= dominant_color[2] <= (CALIB_MAX_LAB[2] + TOLERANCE_BUFFER)

        if is_l_ok and is_a_ok and is_b_ok:
            return True
        else:
            return False

    except Exception as e:
        # print(f"Color check error: {e}") # Commented out to reduce spam
        return False
# ---------------------------------


# --- Main Application ---
# --- MODIFIED: Use new file paths ---
violations_dir = os.path.abspath('violations')
if not os.path.exists(violations_dir): os.makedirs(violations_dir)
daily_csv_path = os.path.join(violations_dir, 'violations_today.csv')
master_csv_path = os.path.join(violations_dir, 'violations_master_log.csv')
# --------------------------------------------------------

# --- NEW: Clear violations folder if it's a new day ---
# This MUST run before load_todays_violations
clear_violations_for_new_day(violations_dir, daily_csv_path)
# -----------------------------------------------------

# --- Create Master Log if it doesn't exist ---
if not os.path.exists(master_csv_path):
    with open(master_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'SourceIdx', 'ID', 'Name', 'ViolationType', 'SnapshotPath'])

# --- Create Daily Log if it doesn't exist (e.g., after clearing) ---
if not os.path.exists(daily_csv_path):
    with open(daily_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'SourceIdx', 'ID', 'Name', 'ViolationType', 'SnapshotPath'])

# --- NEW: Run the Monthly Report Generator ---
# This runs *after* clearing, so it always has the latest master data
generate_monthly_report(master_csv_path, violations_dir)
# -----------------------------------------------

# --- NEW: Run the Source Chooser UI ---
if 'pytest' in sys.modules: # Simple check to skip UI during testing
    MODE = 'video'
    VIDEO_SOURCES = [r"C:\Users\syeda\yolo_project\IMG_6472.MP4"] # Dummy file for testing
    SCREEN_W, SCREEN_H = 1920, 1080 # Mock screen size
else:
    print("Showing source selection window...")
    chooser = SourceChooser()
    # --- MODIFIED: Now we get screen dimensions too ---
    MODE, VIDEO_SOURCES, SCREEN_W, SCREEN_H = chooser.run()

if not VIDEO_SOURCES:
    print("No source selected. Exiting program.")
    sys.exit()
# --- End of Source Chooser ---

# --- DYNAMIC CONFIGURATION (based on choice) ---
if MODE == 'webcam':
    NUM_SOURCES = 1
    # Frame skip is set up top
elif MODE == 'video':
    NUM_SOURCES = len(VIDEO_SOURCES)
    # Frame skip is set up top
# ---------------------------------------------


# --- UI Layout Configuration (COMPACT LAPTOP MODE) ---
# Adjusted for 16-inch laptop screens with scaling
# We force strict height limits to account for taskbars/titlebars

if NUM_SOURCES == 1: # Single View Layout
    LAYOUT_COLS = 1
    LAYOUT_ROWS = 1
    
    # Use a conservative 60% of width to ensure it fits easily
    VIDEO_WIDTH = int(SCREEN_W * 0.60) 
    VIDEO_HEIGHT = int(VIDEO_WIDTH * 9 / 16)
    
    PANEL_WIDTH = 220 
    TOTAL_VIDEO_PANEL_WIDTH = VIDEO_WIDTH + PANEL_WIDTH
    H_SPACING = 10
    V_SPACING = 15
    DASHBOARD_WIDTH = TOTAL_VIDEO_PANEL_WIDTH + H_SPACING * 2
    DASHBOARD_HEIGHT = VIDEO_HEIGHT + V_SPACING * 2 + 40

else: # Multi-View Grid Layout (2 Cols)
    LAYOUT_COLS = 2 
    LAYOUT_ROWS = (NUM_SOURCES + LAYOUT_COLS - 1) // LAYOUT_COLS
    
    # CRITICAL CHANGE: Force a max height that fits on a laptop
    # Subtracting 200px for taskbar + window title bar
    MAX_AVAILABLE_HEIGHT = SCREEN_H - 200
    
    # Calculate allowed height per row
    ROW_HEIGHT = (MAX_AVAILABLE_HEIGHT - (V_SPACING * 3)) // 2
    
    # Ensure video doesn't get too small, but fits
    VIDEO_HEIGHT = max(180, ROW_HEIGHT - 30) # Minimum 180px height
    VIDEO_WIDTH = int(VIDEO_HEIGHT * 16 / 9)
    
    PANEL_WIDTH = 180 # Make panel even smaller for compact view
    
    TOTAL_VIDEO_PANEL_WIDTH = VIDEO_WIDTH + PANEL_WIDTH
    H_SPACING = 10
    V_SPACING = 10
    DASHBOARD_WIDTH = LAYOUT_COLS * (TOTAL_VIDEO_PANEL_WIDTH + H_SPACING) + H_SPACING
    DASHBOARD_HEIGHT = LAYOUT_ROWS * (VIDEO_HEIGHT + 30 + V_SPACING) + V_SPACING + 30

print(f"Screen Size: {SCREEN_W}x{SCREEN_H}")
print(f"Dashboard Calculated Size: {DASHBOARD_WIDTH}x{DASHBOARD_HEIGHT}")
# -----------------------------

print("Loading models...")
person_model = YOLO('yolov8n.pt')

# --- NEW: LOAD THE ID CARD MODEL ---
try:
    id_model = YOLO('id_detector.pt')
    print("Successfully loaded ID Card detector model.")
except Exception as e:
    print(f"Error: Could not load 'id_detector.pt'. Make sure it's in the folder. {e}")
    sys.exit()
# -----------------------------------

# --- NEW: LOAD THE UNIFORM MODEL ---
try:
    uniform_model = YOLO('uniform_detector.pt') # <-- We load the new model
    print("Successfully loaded Uniform detector model.")
except Exception as e:
    print(f"Error: Could not load 'uniform_detector.pt'. Make sure it's in the folder. {e}")
    sys.exit()
# -------------------------------------

yunet_model_path = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(yunet_model_path):
    print(f"Error: Could not find YuNet model at {yunet_model_path}")
    sys.exit()
face_detector = cv2.FaceDetectorYN.create(yunet_model_path, "", (0, 0))

# --- REMOVED: Old face classifier is no longer needed ---
print("All detection models loaded successfully.")

# --- RE-ADDED: DeepFace warm-up ---
try:
    if not os.path.exists(FACE_DATABASE_PATH) or not os.listdir(FACE_DATABASE_PATH):
        print(f"Error: Your 'face_database' folder is empty or missing.")
        print(f"Please create it at {FACE_DATABASE_PATH} and add subfolders for each person (e.g., 'aman').")
        sys.exit()
        
    print("Warming up Face Recognition model...")
    print("(This may take a moment on first run...)")
    # This command finds all faces in the DB and builds the model
    DeepFace.find(
        img_path = np.zeros((100, 100, 3), dtype=np.uint8), # A dummy image
        db_path = FACE_DATABASE_PATH,
        model_name = 'VGG-Face', # You can also try 'Facenet512'
        enforce_detection = False,
        silent = True
    )
    print("Face Recognition model is ready.")
except Exception as e:
    print(f"Error initializing DeepFace: {e}")
    print("Make sure you have created your 'face_database' folder with subfolders and images.")
    sys.exit()
# --- End of warm-up ---


captures = [cv2.VideoCapture(src) for src in VIDEO_SOURCES]
active_sources = [True] * NUM_SOURCES
video_fps = [0] * NUM_SOURCES
wait_delays = [1] * NUM_SOURCES

for i, cap in enumerate(captures):
    if not cap.isOpened():
        print(f"Error: Could not open source: {VIDEO_SOURCES[i]}")
        active_sources[i] = False
    elif MODE == 'video':
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            video_fps[i] = fps
            delay = int(1000 / (fps * VIDEO_PLAYBACK_SPEED_FACTOR))
            wait_delays[i] = max(1, delay)
        else:
            wait_delays[i] = 25

if not any(active_sources):
    print("Error: Could not open any sources. Exiting.")
    sys.exit()


# --- NEW: Load today's violations to prevent duplicate logging ---
logged_violations = load_todays_violations(daily_csv_path) # Reads the *daily* log
# ---------------------------------------------------------------

frame_counts = [0] * NUM_SOURCES
# --- MODIFIED: This will store data for the UI panel ---
last_detected_persons = [[] for _ in range(NUM_SOURCES)] 
# --- DEFINED: Initialize last_known_data here! ---
last_known_data = {} 
# --------------------------------------------------

# --- RE-ADDED: Dictionaries for "Grace Period" ---
uniform_violation_counts = {}
id_card_violation_counts = {}
# -------------------------------------------------
# --- NEW: Dictionary to store identified names per tracker ---
tracker_names = {}
# --- NEW: Dictionary to track last seen frame for cleanup ---
tracker_last_seen_frame = {}
# -----------------------------------------------------------

while True:
    dashboard = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype="uint8")
    title = "AI Compliance System" + (" - Multi-View" if NUM_SOURCES > 1 else "")
    cv2.putText(dashboard, title, (H_SPACING, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    all_sources_finished = True
    max_wait_delay = 1

    for i in range(NUM_SOURCES):
        row = i // LAYOUT_COLS
        col = i % LAYOUT_COLS
        start_x = H_SPACING + col * (TOTAL_VIDEO_PANEL_WIDTH + H_SPACING)
        start_y = V_SPACING + 50 + row * (VIDEO_HEIGHT + 30 + V_SPACING) # Add space for title
        panel_start_x = start_x + VIDEO_WIDTH

        if not active_sources[i]:
            cv2.rectangle(dashboard, (start_x, start_y), (start_x + VIDEO_WIDTH, start_y + VIDEO_HEIGHT), (30, 30, 30), -1)
            cv2.putText(dashboard, f"Source {i+1} Inactive", (start_x + 10, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.rectangle(dashboard, (panel_start_x, start_y), (panel_start_x + PANEL_WIDTH, start_y + VIDEO_HEIGHT), (20, 20, 20), -1)
            cv2.putText(dashboard, f"SOURCE {i+1}", (panel_start_x + 10, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            continue

        ret, frame = captures[i].read()
        if not ret:
            active_sources[i] = False
            continue

        all_sources_finished = False
        frame_counts[i] += 1
        
        # --- NEW "TWO-LOOP" LOGIC ---
        # 1. FAST LOOP (Every Frame): Run the fast person tracker
        # We run this *outside* the FRAME_SKIP block
        results = person_model.track(frame, classes=0, verbose=False, persist=True)
        person_found_in_ai_frame = False
        
        # We need to store the current boxes for drawing
        current_boxes = []
        if results[0].boxes.id is not None:
            person_found_in_ai_frame = True
            for box, tracker_id_tensor in zip(results[0].boxes, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tracker_id = int(tracker_id_tensor.item())
                # --- NEW: Update the last seen frame for this tracker ---
                tracker_last_seen_frame[tracker_id] = frame_counts[i]
                # -------------------------------------------------------
                current_boxes.append((x1, y1, x2, y2, tracker_id))
        
        # 2. SLOW LOOP (Every N Frames): Run the heavy AI models
        if frame_counts[i] % FRAME_SKIP == 0:
            
            # --- BUG FIX: Cleanup now has a time-to-live (TTL) ---
            # This runs *inside* the slow loop
            all_known_trackers = list(tracker_last_seen_frame.keys())
            
            for old_id in all_known_trackers:
                # If a tracker hasn't been seen in 50 frames, remove it
                if (frame_counts[i] - tracker_last_seen_frame.get(old_id, 0)) > CLEANUP_TTL:
                    # print(f"Cleaning up old Tracker {old_id}...")
                    # This person has left the frame, clean up their data
                    last_known_data.pop(old_id, None)
                    uniform_violation_counts.pop(old_id, None)
                    id_card_violation_counts.pop(old_id, None)
                    tracker_names.pop(old_id, None) # Forget their name
                    tracker_last_seen_frame.pop(old_id, None) # Forget when we saw them
            # --------------------------------------------------------

            # Loop over the people we *just* found in the fast loop
            for x1, y1, x2, y2, tracker_id in current_boxes:
                
                # --- UNIFORM CHECK (YOLO + COLOR) ---
                person_roi_bgr = frame[y1:y2, x1:x2] if y1<y2 and x1<x2 else None
                is_uniform_ok = False
                uniform_perc = 0.0 

                if person_roi_bgr is not None and person_roi_bgr.size > 0:
                    uniform_results = uniform_model(person_roi_bgr, classes=0, verbose=False) 
                    if uniform_results[0].boxes:
                        # 1. YOLO Detected Shape
                        uniform_perc = uniform_results[0].boxes.conf[0].item() * 100
                        
                        # 2. Color Verification
                        # --- IMPORTANT: We now use the 50% threshold logic requested ---
                        if uniform_perc > COMPLIANCE_CONFIDENCE_THRESHOLD:
                            if check_color_compliance(person_roi_bgr):
                                is_uniform_ok = True
                            else:
                                is_uniform_ok = False # Wrong color
                        else:
                            is_uniform_ok = False # Confidence too low
                # ---------------------------------------------

                # --- ID CARD CHECK (NEW: Using YOLO Model) ---
                is_id_card_ok = False
                id_perc = 0.0 # This will store the confidence

                if person_roi_bgr is not None and person_roi_bgr.size > 0:
                    id_results = id_model(person_roi_bgr, classes=0, verbose=False) 
                    if id_results[0].boxes:
                        id_perc = id_results[0].boxes.conf[0].item() * 100
                        
                        # --- IMPORTANT: We now use the 50% threshold logic requested ---
                        if id_perc > COMPLIANCE_CONFIDENCE_THRESHOLD:
                             is_id_card_ok = True
                        else:
                             is_id_card_ok = False # Confidence too low
                # ---------------------------------------------
                
                # --- NEW: "IDENTIFY ON FIRST SIGHT" LOGIC (using DeepFace) ---
                person_id = -1
                if tracker_id not in tracker_names:
                    # This is a new person. Run Face ID *once*.
                    print(f"New person (Tracker {tracker_id}) detected. Running Face ID...")
                    person_name = "Unknown" # Default
                    
                    # 1. Use YuNet to find the face *within* the person crop
                    if person_roi_bgr is not None and person_roi_bgr.size > 0:
                        roi_h, roi_w = person_roi_bgr.shape[:2]
                        if roi_w > 0 and roi_h > 0:
                            face_detector.setInputSize((roi_w, roi_h))
                            faces = face_detector.detect(person_roi_bgr)
                            
                            # 2. If YuNet finds a face...
                            if faces[1] is not None:
                                face_data = faces[1][0] # Get the best face
                                face_coords = face_data[0:4]
                                
                                if all(math.isfinite(coord) for coord in face_coords):
                                    fx, fy, fw, fh = map(int, face_coords)
                                    fx, fy = max(0, fx), max(0, fy)
                                    
                                    if fw > 0 and fh > 0:
                                        # 3. Crop the face for DeepFace
                                        face_crop_bgr = person_roi_bgr[fy:fy+fh, fx:fx+fw]
                                        
                                        if face_crop_bgr.size > 0:
                                            try:
                                                # 4. Run DeepFace.find()
                                                dfs = DeepFace.find(
                                                    img_path = face_crop_bgr,
                                                    db_path = FACE_DATABASE_PATH,
                                                    model_name = 'VGG-Face',
                                                    enforce_detection = False, # We already detected
                                                    silent = True
                                                )
                                                # 5. Parse the results
                                                if dfs and not dfs[0].empty:
                                                    identity_path = dfs[0].iloc[0]['identity']
                                                    person_name = os.path.basename(os.path.dirname(identity_path))
                                                    person_id = 0 # We don't have a numeric ID
                                                    
                                            except Exception as e:
                                                pass # Name remains "Unknown"
                    
                    # Store the result (even if it's "Unknown")
                    tracker_names[tracker_id] = (person_name, person_id)
                    print(f"Tracker {tracker_id} identified as: {person_name}")
                else:
                    # This person is already known. Get name from memory.
                    person_name, person_id = tracker_names[tracker_id]
                # --- END OF "IDENTIFY ON FIRST SIGHT" ---
                
                # --- MODIFIED: "GRACE PERIOD" VIOLATION LOGIC ---
                if person_name != "Unknown":
                    
                    # --- Check Uniform ---
                    if is_uniform_ok:
                        uniform_violation_counts[tracker_id] = 0
                    else:
                        uniform_violation_counts[tracker_id] = uniform_violation_counts.get(tracker_id, 0) + 1
                    
                    # --- Check ID Card ---
                    if is_id_card_ok:
                        id_card_violation_counts[tracker_id] = 0
                    else:
                        id_card_violation_counts[tracker_id] = id_card_violation_counts.get(tracker_id, 0) + 1

                    # --- Debugging Print ---
                    if uniform_violation_counts.get(tracker_id, 0) > 0 or id_card_violation_counts.get(tracker_id, 0) > 0:
                            # print(f"DEBUG: Tracker {tracker_id} ({person_name}) Counts - Uniform: {uniform_violation_counts.get(tracker_id, 0)}, ID: {id_card_violation_counts.get(tracker_id, 0)}")
                            pass
                    
                    # --- NOW, check if any count has exceeded the grace period ---
                    v_type = None
                    if uniform_violation_counts.get(tracker_id, 0) > VIOLATION_GRACE_PERIOD:
                        v_type = "Uniform"
                    elif id_card_violation_counts.get(tracker_id, 0) > VIOLATION_GRACE_PERIOD:
                        v_type = "ID Card"
                    
                    if v_type is not None:
                        # --- This is a Confirmed, Loggable Violation ---
                        print(f"CONFIRMED VIOLATION for {person_name}: {v_type}. Attempting to log...")
                        
                        # --- Check if already logged *before* saving ---
                        violation_key = (person_name.lower(), v_type)
                        if violation_key not in logged_violations:
                            # --- This is a NEW violation ---
                            print(f"--- Logged new violation for {person_name}: {v_type}. This person will not be logged for this again. ---")
                            logged_violations.add(violation_key) # Add to in-memory set
                            
                            ts = datetime.now()
                            ts_string_db = ts.strftime("%Y-%m-%d %H:%M:%S") # For CSV/DB
                            ts_string_file = ts.strftime("%Y%m%d_%H%M%S") # For filename
                            path = os.path.join(violations_dir, f"v_src{i+1}_{person_name}_{v_type}_{ts_string_file}.jpg")
                            
                            if frame is not None and frame.size > 0:
                                # 1. SAVE THE SNAPSHOT FIRST
                                snapshot_frame = frame.copy()
                                cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0,0,255), 2)
                                cv2.putText(snapshot_frame, f"VIOLATION: {v_type}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                                cv2.imwrite(path, snapshot_frame)
                                
                                # 2. SAVE TO *BOTH* CSVs
                                log_row = [ts_string_db, i+1, person_id, person_name, v_type, path]
                                with open(daily_csv_path, 'a', newline='') as f:
                                    csv.writer(f).writerow(log_row)
                                with open(master_csv_path, 'a', newline='') as f:
                                    csv.writer(f).writerow(log_row)

                                # 3. SEND WHATSAPP ALERT
                                source_name = f"Source {i+1}" if NUM_SOURCES > 1 else "Webcam"
                                send_whatsapp_violation(person_name, v_type, source_name, path)
                        
                        else:
                            # This is a duplicate violation
                            # print(f"Ignoring duplicate violation for {person_name}: {v_type}")
                            pass
                        
                        # --- IMPORTANT: Reset count after logging ---
                        if v_type == "Uniform": uniform_violation_counts[tracker_id] = 0
                        if v_type == "ID Card": id_card_violation_counts[tracker_id] = 0
                # --- END OF "GRACE PERIOD" LOGIC ---

                # --- NEW: Store the AI results in our "memory" ---
                last_known_data[tracker_id] = {
                    'name': person_name,
                    'is_uniform_ok': is_uniform_ok,
                    'is_id_card_ok': is_id_card_ok,
                    'uniform_perc': uniform_perc,
                    'id_perc': id_perc,
                    'box': (x1, y1, x2, y2) # Store the box from the AI frame
                }
                # ------------------------------------------------
            
        # --- End of SLOW LOOP ---
        
        # --- UI Drawing (Happens Every Frame) ---
        try:
            if frame is not None and frame.size > 0:
                frame_resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                # --- MODIFIED: We will draw our *own* boxes, not from result.plot() ---
                # dashboard[start_y : start_y + VIDEO_HEIGHT, start_x : start_x + VIDEO_WIDTH] = frame_resized
            else: raise ValueError("Frame is empty or invalid")
        except Exception as e:
            cv2.rectangle(dashboard, (start_x, start_y), (start_x + VIDEO_WIDTH, start_y + VIDEO_HEIGHT), (50, 50, 50), -1)
            cv2.putText(dashboard, f"Source {i+1} Error", (start_x + 10, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            active_sources[i] = False
            continue

        cv2.rectangle(dashboard, (panel_start_x, start_y), (panel_start_x + PANEL_WIDTH, start_y + VIDEO_HEIGHT), (30, 30, 30), -1)
        cv2.putText(dashboard, f"SOURCE {i+1}", (panel_start_x + 10, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # --- MODIFIED: UI now uses the fast *current_boxes* and the *last_known_data* ---
        
        # Draw boxes for ALL *currently* tracked people
        orig_h, orig_w = frame.shape[:2]
        if orig_w > 0 and orig_h > 0:
            scale_x, scale_y = VIDEO_WIDTH / orig_w, VIDEO_HEIGHT / orig_h
            
            # Use the boxes from the *fast loop*
            for x1_orig, y1_orig, x2_orig, y2_orig, tracker_id in current_boxes:
                
                # Get the *data* for this tracker from our "memory"
                person_data = last_known_data.get(tracker_id)
                
                if person_data:
                    # We have AI data for this person
                    name = person_data['name']
                    is_uniform_violating = uniform_violation_counts.get(tracker_id, 0) > 0
                    is_id_violating = id_card_violation_counts.get(tracker_id, 0) > 0
                    compliant = (not is_uniform_violating) and (not is_id_violating)
                else:
                    # AI hasn't processed this person yet
                    name = "..." # Show loading/processing
                    compliant = True # Assume compliant until proven otherwise
                
                # Red if violating AND known, Green if compliant AND known, Yellow for Unknown
                color = (0, 0, 255) if (not compliant and name != "Unknown") else ( (0, 255, 0) if name != "Unknown" else (255, 255, 0) ) 
                
                # Use the *current* box coordinates for smooth drawing
                rx1, ry1 = int(x1_orig*scale_x), int(y1_orig*scale_y)
                rx2, ry2 = int(x2_orig*scale_x), int(y2_orig*scale_y)
                rx1_dash, ry1_dash = max(0, start_x + rx1), max(0, start_y + ry1)
                rx2_dash, ry2_dash = min(start_x + VIDEO_WIDTH, start_x + rx2), min(start_y + VIDEO_HEIGHT, start_y + ry2)
                
                if rx1_dash < rx2_dash and ry1_dash < ry2_dash:
                    cv2.rectangle(frame_resized, (rx1, ry1), (rx2, ry2), color, 1) # Draw on the resized frame
                    cv2.putText(frame_resized, name.upper(), (rx1 + 5, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Put the frame with boxes onto the dashboard
        dashboard[start_y : start_y + VIDEO_HEIGHT, start_x : start_x + VIDEO_WIDTH] = frame_resized

        # --- Update Info Panel (shows first person in memory) ---
        if not current_boxes:
            cv2.putText(dashboard, "Searching...", (panel_start_x + 10, start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            # Get data for the *first* tracker ID we see
            first_tracker_id = current_boxes[0][4]
            first_person_data = last_known_data.get(first_tracker_id)

            if not first_person_data:
                # AI hasn't run yet for this person
                cv2.putText(dashboard, "Processing...", (panel_start_x + 10, start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                # We have data, display it
                name = first_person_data['name']
                is_uniform_ok = first_person_data['is_uniform_ok']
                is_id_card_ok = first_person_data['is_id_card_ok']
                uniform_perc = first_person_data['uniform_perc']
                id_perc = first_person_data['id_perc']
                
                uniform_color = (0, 255, 0) if is_uniform_ok else (0, 0, 255)
                id_color = (0, 255, 0) if is_id_card_ok else (0, 0, 255)

                cv2.putText(dashboard, f"Person: {name}", (panel_start_x + 10, start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(dashboard, f"Uniform: {'OK' if is_uniform_ok else 'VIOLATION'}", (panel_start_x + 10, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, uniform_color, 1)
                cv2.putText(dashboard, f" (Conf: {int(uniform_perc)}%)", (panel_start_x + 150, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(dashboard, f"ID Card: {'OK' if is_id_card_ok else 'VIOLATION'}", (panel_start_x + 10, start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 1)
                cv2.putText(dashboard, f" (Conf: {int(id_perc)}%)", (panel_start_x + 150, start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.putText(dashboard, f"People Detected: {len(current_boxes)}", (panel_start_x + 10, start_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # --- End of UI Update ---

        max_wait_delay = max(max_wait_delay, wait_delays[i])

    cv2.imshow('AI Compliance System', dashboard)
    
    # New: Maximize window to handle taskbars
    try:
        # Windows only: maximize window to fill available space
        cv2.setWindowProperty('AI Compliance System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except:
        pass

    if all_sources_finished:
        print("All video sources have finished.")
        cv2.waitKey(0)
        break

    if cv2.waitKey(max_wait_delay) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("\nExiting Program and cleaning up.")
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
os._exit(0)