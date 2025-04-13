import os
import json
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QCheckBox, QFrame, QFileDialog, QMessageBox,
                             QInputDialog)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

# Email Configuration (same as in face_recognition_app.py)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "hotelardilla@gmail.com"  # Replace with your email
SENDER_PASSWORD = "qviw qyaj kwsj hfgd"  # Replace with your app-specific password

class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login to Intelligent Attendance System")
        self.setFixedSize(700, 700)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e0e7ff, stop:1 #f9faff);
                border-radius: 20px;
                border: 1px solid #d1d5db;
            }
            QLabel#titleLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 24px;
                font-weight: 700;
                color: #1f2937;
            }
            QLabel#subtitleLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #6b7280;
            }
            QLabel {
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #1f2937;
            }
            QLineEdit {
                padding: 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background-color: #ffffff;
                font-size: 14px;
                font-family: 'Segoe UI', sans-serif;
                color: #374151;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
                background-color: #f0f7ff;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb);
                color: white;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-family: 'Segoe UI', sans-serif;
                font-weight: 600;
                transition: background 0.3s;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #1d4ed8);
            }
            QPushButton#secondaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6b7280, stop:1 #4b5563);
            }
            QPushButton#secondaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4b5563, stop:1 #374151);
            }
            QCheckBox {
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                color: #374151;
            }
            QFrame#iconFrame {
                background: transparent;
                border: none;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        icon_frame = QFrame()
        icon_frame.setObjectName("iconFrame")
        icon_layout = QVBoxLayout(icon_frame)
        icon_layout.setAlignment(Qt.AlignCenter)
        self.icon_label = QLabel()
        pixmap = QPixmap("icons/app_icon.png")
        if not pixmap.isNull():
            self.icon_label.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.icon_label.setText("IAS")
            self.icon_label.setStyleSheet("font-size: 36px; font-weight: 700; color: #3b82f6;")
        icon_layout.addWidget(self.icon_label)
        layout.addWidget(icon_frame)

        title_label = QLabel("Welcome Back!")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        subtitle_label = QLabel("Login to manage your attendance system")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)

        form_layout = QVBoxLayout()
        form_layout.setSpacing(15)

        username_layout = QHBoxLayout()
        username_icon = QLabel()
        username_icon.setPixmap(QPixmap("icons/user.png").scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.username = QLineEdit()
        self.username.setPlaceholderText("Enter username")
        username_layout.addWidget(username_icon)
        username_layout.addWidget(self.username)
        form_layout.addLayout(username_layout)

        password_layout = QHBoxLayout()
        password_icon = QLabel()
        password_icon.setPixmap(QPixmap("icons/lock.png").scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.password = QLineEdit()
        self.password.setPlaceholderText("Enter password")
        self.password.setEchoMode(QLineEdit.Password)
        password_layout.addWidget(password_icon)
        password_layout.addWidget(self.password)
        form_layout.addLayout(password_layout)

        self.remember = QCheckBox("Remember me")
        form_layout.addWidget(self.remember, alignment=Qt.AlignLeft)

        layout.addLayout(form_layout)

        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        self.login_btn = QPushButton("Login")
        self.register_btn = QPushButton("Register New Admin")
        self.register_btn.setObjectName("secondaryBtn")
        buttons_layout.addWidget(self.login_btn)
        buttons_layout.addWidget(self.register_btn)
        layout.addLayout(buttons_layout)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 12px; color: #dc2626; font-style: italic;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.login_btn.clicked.connect(self.authenticate)
        self.register_btn.clicked.connect(self.register_Admin)
        self.load_credentials()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def send_email(self, to_email, subject, body):
        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
            server.quit()
            self.status_label.setText(f"Email sent to {to_email}")
            self.status_label.setStyleSheet("font-size: 12px; color: #16a34a; font-style: italic;")
        except Exception as e:
            self.status_label.setText(f"Failed to send email: {str(e)}")
            self.status_label.setStyleSheet("font-size: 12px; color: #dc2626; font-style: italic;")

    def authenticate(self):
        username = self.username.text().strip()
        password = self.password.text().strip()

        if not username or not password:
            self.status_label.setText("Please enter both username and password")
            return

        if not os.path.exists('credentials.json'):
            self.status_label.setText("No admins registered!")
            return

        try:
            with open('credentials.json') as f:
                users = json.load(f)
        except Exception as e:
            self.status_label.setText("Error loading credentials")
            return

        hashed = self.hash_password(password)
        user_data = users.get(username)

        if not user_data:
            self.status_label.setText("Invalid username or password")
            return

        # Handle both old and new formats
        if isinstance(user_data, str):
            # Old format: user_data is the hashed password
            if user_data == hashed:
                # Migrate to new format by adding an email field
                email, ok = QInputDialog.getText(self, "Update Profile", "Please enter your email address:")
                if ok and email:
                    email = email.strip()
                    if '@' not in email or '.' not in email:
                        self.status_label.setText("Please enter a valid email address")
                        return
                    users[username] = {
                        'password': user_data,
                        'email': email
                    }
                    try:
                        with open('credentials.json', 'w') as f:
                            json.dump(users, f)
                    except Exception as e:
                        self.status_label.setText("Error updating credentials")
                        return
                else:
                    # If no email provided, use a placeholder
                    users[username] = {
                        'password': user_data,
                        'email': ''
                    }
                    try:
                        with open('credentials.json', 'w') as f:
                            json.dump(users, f)
                    except Exception as e:
                        self.status_label.setText("Error updating credentials")
                        return
                # Proceed with login
                if self.remember.isChecked():
                    with open('remember.me', 'w') as f:
                        f.write(username)
                else:
                    if os.path.exists('remember.me'):
                        os.remove('remember.me')
                self.status_label.setText("")
                # Send login notification email if email exists
                if email:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    subject = "Login Notification"
                    body = f"Dear {username},\n\nYou have successfully logged in to the Intelligent Attendance System at {timestamp}.\n\nBest regards,\nIntelligent Attendance System"
                    self.send_email(email, subject, body)
                self.accept()
            else:
                self.status_label.setText("Invalid username or password")
        elif isinstance(user_data, dict):
            # New format: user_data is a dictionary
            if user_data.get('password') == hashed:
                if self.remember.isChecked():
                    with open('remember.me', 'w') as f:
                        f.write(username)
                else:
                    if os.path.exists('remember.me'):
                        os.remove('remember.me')
                self.status_label.setText("")
                # Send login notification email if email exists
                email = user_data.get('email')
                if email:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    subject = "Login Notification"
                    body = f"Dear {username},\n\nYou have successfully logged in to the Intelligent Attendance System at {timestamp}.\n\nBest regards,\nIntelligent Attendance System"
                    self.send_email(email, subject, body)
                self.accept()
            else:
                self.status_label.setText("Invalid username or password")
        else:
            self.status_label.setText("Invalid credentials format")

    def register_Admin(self):
        username, ok1 = QInputDialog.getText(self, "Registration", "Enter username:")
        if not ok1 or not username:
            return

        password, ok2 = QInputDialog.getText(self, "Registration", "Enter password:", QLineEdit.Password)
        if not ok2 or not password:
            return

        email, ok3 = QInputDialog.getText(self, "Registration", "Enter email address:")
        if not ok3 or not email:
            return

        username = username.strip()
        password = password.strip()
        email = email.strip()

        # Basic email validation
        if '@' not in email or '.' not in email:
            self.status_label.setText("Please enter a valid email address")
            return

        if len(username) < 4 or len(password) < 6:
            self.status_label.setText("Username must be 4+ chars, password 6+ chars")
            return

        users = {}
        if os.path.exists('credentials.json'):
            try:
                with open('credentials.json') as f:
                    users = json.load(f)
            except Exception as e:
                self.status_label.setText("Error accessing credentials file")
                return

        if username in users:
            self.status_label.setText("Username already exists!")
            return

        # Store username, hashed password, and email
        users[username] = {
            'password': self.hash_password(password),
            'email': email
        }
        try:
            with open('credentials.json', 'w') as f:
                json.dump(users, f)
            QMessageBox.information(self, "Success", "Admin registered successfully!")
            self.status_label.setText("Registration successful! Please login.")
            self.status_label.setStyleSheet("font-size: 12px; color: #16a34a; font-style: italic;")
            # Send registration confirmation email
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            subject = "Registration Confirmation"
            body = f"Dear {username},\n\nYou have been successfully registered in the Intelligent Attendance System at {timestamp}.\nYour username is: {username}\n\nPlease keep this information safe.\n\nBest regards,\nIntelligent Attendance System"
            self.send_email(email, subject, body)
        except Exception as e:
            self.status_label.setText("Error saving credentials")

    def accept(self):
        self.credentials = self.get_credentials()
        super().accept()

    def get_credentials(self):
        return self.username.text(), self.password.text()

    def load_credentials(self):
        if os.path.exists('remember.me'):
            try:
                with open('remember.me') as f:
                    username = f.read().strip()
                    self.username.setText(username)
                    self.remember.setChecked(True)
            except Exception as e:
                self.status_label.setText("Error loading saved credentials")
        return self.get_credentials()