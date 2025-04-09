import cv2
import face_recognition
import os
import numpy as np

# Configuration
KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.5
MODEL = 'hog'
RESIZE_FACTOR = 0.25

# Global lists for known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Load face encodings from images in known_faces folder."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: Directory '{KNOWN_FACES_DIR}' not found.")
        return
    
    print(f"Loading faces from '{KNOWN_FACES_DIR}'...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"Loaded '{name}' from '{image_path}'")
                else:
                    print(f"No face detected in '{image_path}'")
            except Exception as e:
                print(f"Error loading '{image_path}': {str(e)}")
    
    print(f"Total known faces loaded: {len(known_face_names)}")

def recognize_faces():
    """Run face recognition using the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        print(f"Detected {len(face_locations)} faces in frame")
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top = int(top / RESIZE_FACTOR)
            right = int(right / RESIZE_FACTOR)
            bottom = int(bottom / RESIZE_FACTOR)  # Fixed 'custom' to 'bottom'
            left = int(left / RESIZE_FACTOR)
            
            name = "Unknown"
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    print(f"Recognized: {name}")
                else:
                    print("No match found for detected face.")
            else:
                print("No known faces to compare against.")
            
            # Draw rectangle and name on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Ensure known_faces directory exists
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created directory: {KNOWN_FACES_DIR}")
    
    # Load known faces
    load_known_faces()
    
    # Start recognition
    if known_face_encodings:
        recognize_faces()
    else:
        print("No known faces loaded. Please add images to 'known_faces' and try again.")