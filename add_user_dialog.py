import os
import cv2
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer

class AddUserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New User")
        self.setFixedSize(800, 600)
        self.setStyleSheet("""
            QDialog { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e0e7ff, stop:1 #f9faff); border: 1px solid #d1d5db; border-radius: 10px; }
            QLabel { font-family: 'Segoe UI', sans-serif; font-size: 12pt; color: #1f2937; }
            QLineEdit { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-size: 12pt; font-family: 'Segoe UI', sans-serif; color: #374151; }
            QLineEdit:focus { border: 2px solid #3b82f6; background-color: #f0f7ff; }
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 12pt; font-family: 'Segoe UI', sans-serif; font-weight: 600; transition: background 0.3s; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #1d4ed8); }
            QPushButton#secondaryBtn { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6b7280, stop:1 #4b5563); }
            QPushButton#secondaryBtn:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4b5563, stop:1 #374151); }
            QFrame#previewFrame { border: 2px dashed #d1d5db; border-radius: 10px; background-color: #f3f4f6; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        name_label = QLabel("Full Name:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter user's full name")
        
        preview_label = QLabel("Camera Preview:")
        self.preview_frame = QLabel()
        self.preview_frame.setFixedSize(320, 240)
        self.preview_frame.setAlignment(Qt.AlignCenter)
        self.preview_frame.setObjectName("previewFrame")
        self.preview_frame.setText("Camera preview will appear here")
        
        buttons_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.setIcon(QIcon("icons/camera.png"))
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setObjectName("secondaryBtn")
        buttons_layout.addWidget(self.capture_btn)
        buttons_layout.addWidget(self.browse_btn)
        
        self.status_label = QLabel("Ready to add new user")
        self.status_label.setStyleSheet("color: #6b7280; font-style: italic; font-size: 11pt;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        action_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save User")
        self.save_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("secondaryBtn")
        action_layout.addWidget(self.cancel_btn)
        action_layout.addWidget(self.save_btn)
        
        layout.addWidget(name_label)
        layout.addWidget(self.name_input)
        layout.addWidget(preview_label)
        layout.addWidget(self.preview_frame, 1, Qt.AlignCenter)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
        layout.addLayout(action_layout)
        
        self.cancel_btn.clicked.connect(self.reject)
        self.browse_btn.clicked.connect(self.browse_image)
        self.capture_btn.clicked.connect(self.capture_image)
        self.save_btn.clicked.connect(self.accept)
        
        self.image_path = None
        self.captured_image = None
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Failed to open webcam")
            self.status_label.setStyleSheet("color: #dc2626; font-size: 11pt;")
            self.capture_btn.setEnabled(False)
        else:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_preview)
            self.timer.start(30)

    def update_preview(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.preview_frame.setPixmap(pixmap.scaled(self.preview_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_image = frame
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.preview_frame.setPixmap(pixmap.scaled(self.preview_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.status_label.setText("Image captured successfully")
            self.status_label.setStyleSheet("color: #16a34a; font-size: 11pt;")
            self.save_btn.setEnabled(True)
            self.timer.stop()  # Stop live preview after capture
        else:
            self.status_label.setText("Failed to capture image")
            self.status_label.setStyleSheet("color: #dc2626; font-size: 11pt;")

    def browse_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_name:
            pixmap = QPixmap(file_name)
            if not pixmap.isNull():
                self.preview_frame.setPixmap(pixmap.scaled(self.preview_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.image_path = file_name
                self.captured_image = None  # Clear captured image if browsing
                self.status_label.setText("Image loaded successfully")
                self.status_label.setStyleSheet("color: #16a34a; font-size: 11pt;")
                self.save_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to load image")
                self.status_label.setStyleSheet("color: #dc2626; font-size: 11pt;")

    def accept(self):
        if self.captured_image is not None:
            self.image_path = os.path.join('known_faces', f"{self.name_input.text()}_temp.jpg")
            cv2.imwrite(self.image_path, self.captured_image)
        super().accept()

    def reject(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().reject()