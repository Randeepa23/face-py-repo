import os
import csv
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
                             QComboBox, QPushButton, QMessageBox, QDialogButtonBox)
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp
from add_user_dialog import AddUserDialog

class StudentRegistrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Student Registration")
        self.setFixedSize(500, 400)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.fields = {}
        grid = QGridLayout()
        
        # Student ID
        grid.addWidget(QLabel("Student ID:"), 0, 0)
        self.fields['id'] = QLineEdit()
        self.fields['id'].setValidator(QRegExpValidator(QRegExp("[0-9]{9}")))
        grid.addWidget(self.fields['id'], 0, 1)

        # Name
        grid.addWidget(QLabel("Full Name:"), 1, 0)
        self.fields['name'] = QLineEdit()
        grid.addWidget(self.fields['name'], 1, 1)

        # Email
        grid.addWidget(QLabel("Email:"), 2, 0)
        self.fields['email'] = QLineEdit()
        self.fields['email'].setValidator(QRegExpValidator(QRegExp("[^@]+@[^@]+\.[^@]+")))
        grid.addWidget(self.fields['email'], 2, 1)

        # Course
        grid.addWidget(QLabel("Course:"), 3, 0)
        self.fields['course'] = QComboBox()
        self.fields['course'].addItems([
            "Computer Science",
            "Computer Engineering",
            "Software Engineering",
            "Electrical Engineering",
            "Mechanical Engineering",
            "Civil Engineering",
            "Law",
            "Data Science",
            "Management",
            "Medicine"
        ])
        grid.addWidget(self.fields['course'], 3, 1)

        layout.addLayout(grid)
        
        # Face Enrollment
        self.face_btn = QPushButton("Enroll Face")
        self.face_btn.clicked.connect(self.enroll_face)
        layout.addWidget(self.face_btn)
        
        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.validate)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def enroll_face(self):
        dialog = AddUserDialog()
        if dialog.exec_() == QDialog.Accepted:
            self.face_image = dialog.image_path
            QMessageBox.information(self, "Success", "Face enrolled successfully!")

    def validate(self):
        if not all(self.fields[f].text() for f in ['id', 'name', 'email']):
            QMessageBox.warning(self, "Error", "All fields are required!")
            return
        
        if not hasattr(self, 'face_image'):
            QMessageBox.warning(self, "Error", "Please enroll face!")
            return

        # Save to students.csv
        data = {
            'id': self.fields['id'].text(),
            'name': self.fields['name'].text(),
            'email': self.fields['email'].text(),
            'course': self.fields['course'].currentText(),
            'face_image': self.face_image
        }

        file_exists = os.path.exists('students.csv')
        with open('students.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        
        # Move face image to known_faces directory
        new_path = os.path.join('known_faces', f"{data['name']}.jpg")
        os.rename(self.face_image, new_path)
        
        QMessageBox.information(self, "Success", "Student registered successfully!")
        self.accept()
        
    def get_result(self):
        return self.result