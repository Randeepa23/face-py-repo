import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                            QWidget, QPushButton, QDialog, QFileDialog, QMessageBox, QListWidget)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
import os
import face_recognition

# Configuration
KNOWN_FACES_DIR = 'known_faces'

class AddUserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New User")
        self.setFixedSize(400, 500)
        
        layout = QVBoxLayout(self)
        self.name_label = QLabel("Full Name:")  # Fixed: Renamed to avoid overwriting
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter user's full name")
        
        self.preview_frame = QLabel("Camera preview will appear here")
        self.preview_frame.setFixedSize(320, 240)
        self.preview_frame.setAlignment(Qt.AlignCenter)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_image)
        
        self.save_btn = QPushButton("Save User")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.accept)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.cancel_btn)
        buttons_layout.addWidget(self.save_btn)
        
        layout.addWidget(self.name_label)  # Fixed: Use separate label variable
        layout.addWidget(self.name_input)
        layout.addWidget(self.preview_frame)
        layout.addWidget(self.browse_btn)
        layout.addLayout(buttons_layout)
        
        self.image_path = None
        print("AddUserDialog initialized")

    def browse_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)"
        )
        if file_name:
            pixmap = QPixmap(file_name)
            if not pixmap.isNull():
                self.preview_frame.setPixmap(pixmap.scaled(
                    self.preview_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                self.image_path = file_name
                self.save_btn.setEnabled(True)
                print(f"Image selected: {file_name}")
            else:
                print(f"Failed to load image: {file_name}")

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Attendance System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.users_tab = QWidget()
        self.setup_users_tab()
        self.main_layout.addWidget(self.users_tab)
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def setup_users_tab(self):
        layout = QVBoxLayout(self.users_tab)
        
        self.add_user_button = QPushButton("Add New User")
        self.add_user_button.setIcon(QIcon("icons/add_user.png"))
        self.add_user_button.clicked.connect(self.add_new_user)
        
        self.user_list = QListWidget()
        layout.addWidget(self.add_user_button)
        layout.addWidget(self.user_list)
        
        print("Users tab setup complete")

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
            print(f"Created directory: {KNOWN_FACES_DIR}")
            return
        
        print(f"Loading known faces from '{KNOWN_FACES_DIR}'...")
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        print(f"Loaded '{name}' from '{image_path}'")
                    else:
                        print(f"No face detected in '{image_path}'")
                except Exception as e:
                    print(f"Error loading '{image_path}': {str(e)}")
        print(f"Total known faces loaded: {len(self.known_face_names)}")
        self.load_user_list()

    def load_user_list(self):
        self.user_list.clear()
        for name in self.known_face_names:
            self.user_list.addItem(name)
        print(f"User list updated with {len(self.known_face_names)} names")

    def add_new_user(self):
        print("Add New User button clicked")
        dialog = AddUserDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name = dialog.name_input.text()
            if not name:
                print("No name entered, aborting")
                QMessageBox.warning(self, "Error", "Please enter a name.")
                return
            
            if dialog.image_path:
                new_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                try:
                    pixmap = QPixmap(dialog.image_path)
                    if pixmap.save(new_image_path, "JPG"):
                        print(f"Saved new user image to '{new_image_path}'")
                        self.load_known_faces()
                    else:
                        print(f"Failed to save image to '{new_image_path}'")
                        QMessageBox.critical(self, "Error", "Failed to save the image.")
                except Exception as e:
                    print(f"Error saving image: {str(e)}")
                    QMessageBox.critical(self, "Error", f"Error saving image: {str(e)}")
            else:
                print("No image selected, aborting")
                QMessageBox.warning(self, "Error", "Please select an image.")
        else:
            print("Dialog cancelled")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())