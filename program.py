import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import time # To add a small delay

# --- Configuration ---
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_FILE = 'attendance.csv'
TOLERANCE = 0.5  # Lower value means stricter matching (0.4-0.6 is common)
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # 'hog' is faster on CPU, 'cnn' is more accurate but slower (requires dlib compiled with CUDA for GPU)
RESIZE_FACTOR = 0.25 # Resize frame for faster processing (e.g., 0.25 means 1/4 size)
RECOGNITION_DELAY = 5 # Seconds to wait before marking the same person again

# --- Load Known Faces ---
print("Loading known faces...")
known_face_encodings = []
known_face_names = []
last_recognition_time = {} # Dictionary to store the last time each person was recognized

# Check if the known_faces directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"ERROR: Directory not found: {KNOWN_FACES_DIR}")
    print("Please create the directory and add subfolders with names and images.")
    exit()

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.isdir(person_dir): # Check if it's a directory
        print(f"Processing images for: {name}")
        image_count = 0
        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            # Check if it's an image file (basic check)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = face_recognition.load_image_file(filepath)
                    # Get encodings (might be multiple faces in one image)
                    # Use the first encoding found
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        encoding = encodings[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                        image_count += 1
                        print(f"  - Loaded encoding from {filename}")
                    else:
                        print(f"  - WARNING: No face found in {filename}")

                except Exception as e:
                    print(f"  - ERROR loading {filename}: {e}")
        if image_count == 0:
             print(f"  - WARNING: No images with detectable faces found for {name}")
        # Initialize last recognition time for this person
        last_recognition_time[name] = 0

if not known_face_encodings:
    print("ERROR: No known face encodings loaded. Please check the 'known_faces' directory structure and images.")
    exit()

print(f"Loaded {len(known_face_encodings)} known face encodings.")


# --- Initialize Attendance File ---
def mark_attendance(name):
    """Marks attendance and ensures no duplicate entry for the same day."""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    today_entries = set() # Names marked today

    # Create file and header if it doesn't exist
    if not os.path.isfile(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Time', 'Date'])
            print(f"Created '{ATTENDANCE_FILE}' with header.")

    # Read existing entries for today to avoid duplicates
    try:
        with open(ATTENDANCE_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if len(row) >= 3 and row[2] == current_date:
                    today_entries.add(row[0])
    except FileNotFoundError:
        # File might have been deleted after the initial check, proceed to write.
        pass
    except Exception as e:
        print(f"Error reading attendance file: {e}")
        # Decide how to handle: maybe exit or try to continue writing

    # Check if already marked today
    if name not in today_entries:
        # Check recognition delay
        current_timestamp = time.time()
        if current_timestamp - last_recognition_time.get(name, 0) > RECOGNITION_DELAY:
            try:
                with open(ATTENDANCE_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, current_time, current_date])
                    print(f"Attendance Marked: {name} at {current_time} on {current_date}")
                    last_recognition_time[name] = current_timestamp # Update last recognition time
                    return True # Attendance marked
            except Exception as e:
                print(f"Error writing to attendance file: {e}")
        else:
            print(f"Skipping {name}: Recognized within the last {RECOGNITION_DELAY} seconds.")

    else:
        print(f"{name} already marked today ({current_date}).")

    return False # Attendance not marked (duplicate or too recent)

# --- Initialize Video Capture ---
print("Starting video stream...")
video_capture = cv2.VideoCapture(0) # 0 for default webcam

if not video_capture.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Video stream started. Press 'q' to quit.")

# --- Main Loop ---
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Resize frame of video for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    # Alternatively: rgb_small_frame = small_frame[:, :, ::-1]


    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0: # Check if there are any known faces to compare against
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # Attempt to mark attendance
                mark_attendance(name)


        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was resized
        top = int(top / RESIZE_FACTOR)
        right = int(right / RESIZE_FACTOR)
        bottom = int(bottom / RESIZE_FACTOR)
        left = int(left / RESIZE_FACTOR)

        # Draw a box around the face
        if name == "Unknown":
            color = (0, 0, 255) # Red for Unknown
        else:
            color = (0, 255, 0) # Green for Known

        cv2.rectangle(frame, (left, top), (right, bottom), color, FRAME_THICKNESS)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), FONT_THICKNESS)

    # Display the resulting image
    cv2.imshow('Video Face Recognition Attendance (Press q to quit)', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# --- Release handle to the webcam and close windows ---
video_capture.release()
cv2.destroyAllWindows()
print("Resources released.")