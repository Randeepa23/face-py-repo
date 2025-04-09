import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                             QWidget, QListWidget, QPushButton, QGridLayout, QLineEdit,
                             QTextEdit, QGroupBox, QSpinBox, QTableView, QHeaderView,
                             QFrame, QSplitter, QComboBox, QDialog, QFileDialog,
                             QProgressBar, QMessageBox, QTabWidget, QSizePolicy,
                             QGraphicsDropShadowEffect, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QStandardItemModel, QStandardItem, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QTimer, QSize, QMargins
from PyQt5.QtChart import QChart, QChartView, QLineSeries
import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime, timedelta
import time
from sklearn.cluster import DBSCAN
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

# --- Configuration ---
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_FILE = 'attendance.csv'
TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
MODEL = 'hog'
RESIZE_FACTOR = 0.25
RECOGNITION_DELAY = 86400  # 24 hours in seconds

# --- Icon Paths (Replace with your actual icon file paths) ---
ADD_USER_ICON = "icons/add_user.png"
REMOVE_USER_ICON = "icons/remove_user.png"
WEBCAM_ICON = "icons/webcam.png"
MODEL_ICON = "icons/model.png"
FACES_ICON = "icons/faces.png"
SETTINGS_ICON = "icons/settings.png"
EXPORT_ICON = "icons/export.png"
DASHBOARD_ICON = "icons/dashboard.png"
ATTENDANCE_ICON = "icons/attendance.png"
USERS_ICON = "icons/users.png"

# Initialize global variables
known_face_encodings = []
known_face_names = []

class AddUserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New User")
        self.setFixedSize(400, 500)
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
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #d1d5db; border-radius: 5px; background-color: #f3f4f6; text-align: center; font-family: 'Segoe UI', sans-serif; font-size: 10pt; } QProgressBar::chunk { background-color: #3b82f6; border-radius: 4px; }")
        
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
        self.save_btn.clicked.connect(self.accept)
        
        self.image_path = None

    def browse_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_name:
            pixmap = QPixmap(file_name)
            if not pixmap.isNull():
                self.preview_frame.setPixmap(pixmap.scaled(self.preview_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.image_path = file_name
                self.status_label.setText("Image loaded successfully")
                self.status_label.setStyleSheet("color: #16a34a; font-size: 11pt;")
                self.save_btn.setEnabled(True)
            else:
                self.status_label.setText("Failed to load image")
                self.status_label.setStyleSheet("color: #dc2626; font-size: 11pt;")

class StatusWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f9fafb); border-radius: 12px; border: 1px solid #e5e7eb; } QLabel { font-family: 'Segoe UI', sans-serif; color: #1f2937; } QLabel#statusValue { font-weight: 600; font-size: 16px; } QLabel#statusTitle { font-size: 12px; color: #6b7280; }")
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.setGraphicsEffect(shadow)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(8)
        
        self.title_label = QLabel()
        self.title_label.setObjectName("statusTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        self.value_label = QLabel()
        self.value_label.setObjectName("statusValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        
        self.layout.addWidget(self.value_label)
        self.layout.addWidget(self.title_label)
    
    def set_data(self, title, value, color="#1f2937"):
        self.title_label.setText(title)
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color: {color}; font-weight: 600; font-size: 16px; font-family: 'Segoe UI', sans-serif;")

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Attendance System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setWindowIcon(QIcon("icons/app_icon.png"))
        self.setStyleSheet(self.load_stylesheet())
        
        # Initialize attendance_model here
        self.attendance_model = QStandardItemModel(0, 3)
        self.attendance_model.setHorizontalHeaderLabels(["Name", "Time", "Date"])
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setStyleSheet("QTabWidget::pane { border: none; background: #f3f4f6; } QTabBar::tab { padding: 12px 24px; margin: 0 4px; font-size: 14px; font-family: 'Segoe UI', sans-serif; font-weight: 600; min-width: 120px; border-top-left-radius: 8px; border-top-right-radius: 8px; background: #e5e7eb; color: #4b5563; transition: background 0.3s, color 0.3s; } QTabBar::tab:selected { background: #3b82f6; color: white; } QTabBar::tab:hover:!selected { background: #f3f4f6; color: #1f2937; }")
        
        self.dashboard_tab = QWidget()
        self.attendance_tab = QWidget()
        self.users_tab = QWidget()
        self.settings_tab = QWidget()
        
        self.tabs.addTab(self.dashboard_tab, QIcon(DASHBOARD_ICON), "Dashboard")
        self.tabs.addTab(self.attendance_tab, QIcon(ATTENDANCE_ICON), "Attendance")
        self.tabs.addTab(self.users_tab, QIcon(USERS_ICON), "Users")
        self.tabs.addTab(self.settings_tab, QIcon(SETTINGS_ICON), "Settings")
        
        self.main_layout.addWidget(self.tabs)
        
        self.setup_dashboard_tab()
        self.setup_attendance_tab()
        self.setup_users_tab()
        self.setup_settings_tab()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            QMessageBox.critical(self, "Error", "Could not open webcam. Check connection and permissions.")
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        self.greeting_timer = QTimer(self)
        self.greeting_timer.setSingleShot(True)
        self.greeting_timer.timeout.connect(self.hide_greeting)
        
        self.last_recognition_time_ui = {}
        self.load_known_faces()
        self.load_attendance_records()  # Load data after model is initialized
        print("App initialized")

    def setup_dashboard_tab(self):
        layout = QVBoxLayout(self.dashboard_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        title_layout = QVBoxLayout()
        title = QLabel("Attendance Dashboard")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        subtitle = QLabel("Real-time facial recognition attendance monitoring")
        subtitle.setStyleSheet("font-size: 16px; color: #6b7280; font-family: 'Segoe UI', sans-serif;")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addLayout(title_layout)
        
        main_content = QHBoxLayout()
        
        video_panel = QFrame()
        video_panel.setFrameShape(QFrame.StyledPanel)
        video_panel.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f9fafb); border-radius: 15px; border: 1px solid #e5e7eb; }")
        shadow = QGraphicsDropShadowEffect(video_panel)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        video_panel.setGraphicsEffect(shadow)
        
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(20, 20, 20, 20)
        
        video_header = QLabel("Live Recognition Feed")
        video_header.setStyleSheet("font-size: 18px; font-weight: 600; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        video_header.setAlignment(Qt.AlignCenter)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #e5e7eb; border-radius: 10px; background-color: #f9fafb;")
        
        self.greeting_label = QLabel("")
        self.greeting_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #16a34a; font-family: 'Segoe UI', sans-serif; background: #f0fdf4; padding: 10px; border-radius: 8px; border: 1px solid #bbf7d0;")
        self.greeting_label.setAlignment(Qt.AlignCenter)
        self.greeting_label.setVisible(False)
        
        video_footer = QLabel("System actively monitoring for registered users")
        video_footer.setStyleSheet("font-size: 13px; color: #6b7280; font-family: 'Segoe UI', sans-serif;")
        video_footer.setAlignment(Qt.AlignCenter)
        
        # New: User Activity Log
        activity_frame = QFrame()
        activity_frame.setFrameShape(QFrame.StyledPanel)
        activity_frame.setStyleSheet("QFrame { background: #ffffff; border-radius: 10px; border: 1px solid #e5e7eb; }")
        activity_layout = QVBoxLayout(activity_frame)
        activity_header = QLabel("Recent Activity Log")
        activity_header.setStyleSheet("font-size: 14px; font-weight: 600; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet("QListWidget { border: none; background: transparent; font-family: 'Segoe UI', sans-serif; color: #374151; } QListWidget::item { padding: 8px; font-size: 13px; border-bottom: 1px solid #f3f4f6; }")
        self.activity_list.setMaximumHeight(150)  # Limit height to keep it compact
        activity_layout.addWidget(activity_header)
        activity_layout.addWidget(self.activity_list)
        
        video_layout.addWidget(video_header)
        video_layout.addWidget(self.video_label, 1)
        video_layout.addWidget(self.greeting_label)
        video_layout.addWidget(video_footer)
        video_layout.addWidget(activity_frame)  # Add activity log below video
        
        status_panel = QVBoxLayout()
        status_panel.setSpacing(20)
        
        status_row1 = QHBoxLayout()
        self.status_webcam = StatusWidget()
        self.status_webcam.set_data("Webcam Status", "Connecting...", "#dc2626")
        self.status_faces = StatusWidget()
        self.status_faces.set_data("Known Users", str(len(known_face_names)), "#3b82f6")
        status_row1.addWidget(self.status_webcam)
        status_row1.addWidget(self.status_faces)
        
        status_row2 = QHBoxLayout()
        self.status_today = StatusWidget()
        self.status_today.set_data("Today's Attendance", "0", "#16a34a")
        self.status_recognition = StatusWidget()
        self.status_recognition.set_data("Recognition Model", MODEL, "#8b5cf6")
        status_row2.addWidget(self.status_today)
        status_row2.addWidget(self.status_recognition)
        
        trend_frame = QFrame()
        trend_frame.setFrameShape(QFrame.StyledPanel)
        trend_frame.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f9fafb); border-radius: 12px; border: 1px solid #e5e7eb; }")
        shadow = QGraphicsDropShadowEffect(trend_frame)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        trend_frame.setGraphicsEffect(shadow)
        
        trend_layout = QVBoxLayout(trend_frame)
        trend_header = QLabel("Attendance Trends (Last 7 Days)")
        trend_header.setStyleSheet("font-size: 16px; font-weight: 600; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.trend_chart = QChart()
        self.trend_chart.setTitle("Attendance Over Time")
        self.trend_series = QLineSeries()
        self.trend_chart.addSeries(self.trend_series)
        self.trend_chart.createDefaultAxes()
        self.trend_view = QChartView(self.trend_chart)
        self.trend_view.setMinimumHeight(200)
        trend_layout.addWidget(trend_header)
        trend_layout.addWidget(self.trend_view)
        
        status_panel.addLayout(status_row1)
        status_panel.addLayout(status_row2)
        status_panel.addWidget(trend_frame, 1)
        
        main_content.addWidget(video_panel, 3)
        main_content.addLayout(status_panel, 2)
        
        layout.addLayout(main_content, 1)
        self.update_trends()
        print("Dashboard tab setup complete")

    def setup_attendance_tab(self):
        layout = QVBoxLayout(self.attendance_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        header_layout = QHBoxLayout()
        title = QLabel("Attendance Records")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        export_btn = QPushButton("Export Data")
        export_btn.setIcon(QIcon(EXPORT_ICON))
        export_btn.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #16a34a, stop:1 #15803d); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-family: 'Segoe UI', sans-serif; font-weight: 600; transition: background 0.3s; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #15803d, stop:1 #166534); }")
        export_btn.clicked.connect(self.export_data)
        
        date_filter = QComboBox()
        date_filter.addItems(["All Dates", "Today", "Yesterday", "This Week", "This Month"])
        date_filter.setStyleSheet("QComboBox { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; min-width: 150px; font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #374151; } QComboBox:hover { border: 1px solid #3b82f6; } QComboBox::drop-down { border: none; width: 30px; } QComboBox::down-arrow { image: url(icons/down_arrow.png); width: 12px; height: 12px; }")
        
        header_layout.addWidget(title)
        header_layout.addWidget(spacer)
        header_layout.addWidget(date_filter)
        header_layout.addWidget(export_btn)
        
        table_frame = QFrame()
        table_frame.setFrameShape(QFrame.StyledPanel)
        table_frame.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f9fafb); border-radius: 12px; border: 1px solid #e5e7eb; }")
        shadow = QGraphicsDropShadowEffect(table_frame)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        table_frame.setGraphicsEffect(shadow)
        
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(15, 15, 15, 15)
        
        self.attendance_table = QTableView()
        self.attendance_table.setModel(self.attendance_model)
        self.attendance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.attendance_table.setStyleSheet("QTableView { border: none; background: transparent; font-family: 'Segoe UI', sans-serif; color: #374151; selection-background-color: #eff6ff; selection-color: #1f2937; } QHeaderView::section { background: #f9fafb; padding: 12px; border: none; border-bottom: 1px solid #e5e7eb; font-weight: 600; font-size: 14px; color: #1f2937; font-family: 'Segoe UI', sans-serif; } QTableView::item { padding: 12px; border-bottom: 1px solid #f3f4f6; font-size: 13px; }")
        
        table_layout.addWidget(self.attendance_table)
        
        layout.addLayout(header_layout)
        layout.addWidget(table_frame, 1)
        print("Attendance tab setup complete")

    def setup_users_tab(self):
        layout = QVBoxLayout(self.users_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        header_layout = QHBoxLayout()
        title = QLabel("User Management")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.add_user_button = QPushButton("Add New User")
        self.add_user_button.setIcon(QIcon(ADD_USER_ICON))
        self.add_user_button.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-family: 'Segoe UI', sans-serif; font-weight: 600; transition: background 0.3s; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #1d4ed8); }")
        
        header_layout.addWidget(title)
        header_layout.addWidget(spacer)
        header_layout.addWidget(self.add_user_button)
        
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        user_list_frame = QFrame()
        user_list_frame.setFrameShape(QFrame.StyledPanel)
        user_list_frame.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f9fafb); border-radius: 12px; border: 1px solid #e5e7eb; }")
        shadow = QGraphicsDropShadowEffect(user_list_frame)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        user_list_frame.setGraphicsEffect(shadow)
        
        user_list_layout = QVBoxLayout(user_list_frame)
        user_list_layout.setContentsMargins(15, 15, 15, 15)
        
        user_search = QLineEdit()
        user_search.setPlaceholderText("Search users...")
        user_search.setStyleSheet("QLineEdit { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #f9fafb; font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #374151; margin-bottom: 15px; } QLineEdit:focus { border: 2px solid #3b82f6; background-color: #f0f7ff; }")
        
        self.user_list = QListWidget()
        self.user_list.setStyleSheet("QListWidget { border: none; background: transparent; font-family: 'Segoe UI', sans-serif; color: #374151; } QListWidget::item { border-bottom: 1px solid #f3f4f6; padding: 14px; font-size: 14px; } QListWidget::item:selected { background: #eff6ff; color: #1f2937; border-radius: 6px; }")
        
        user_list_layout.addWidget(user_search)
        user_list_layout.addWidget(self.user_list)
        
        user_detail_frame = QFrame()
        user_detail_frame.setFrameShape(QFrame.StyledPanel)
        user_detail_frame.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f9fafb); border-radius: 12px; border: 1px solid #e5e7eb; }")
        shadow = QGraphicsDropShadowEffect(user_detail_frame)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        user_detail_frame.setGraphicsEffect(shadow)
        
        user_detail_layout = QVBoxLayout(user_detail_frame)
        user_detail_layout.setContentsMargins(20, 20, 20, 20)
        
        detail_header = QLabel("User Details")
        detail_header.setStyleSheet("font-size: 20px; font-weight: 600; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        
        user_photo_frame = QFrame()
        user_photo_frame.setFixedSize(200, 200)
        user_photo_frame.setStyleSheet("background-color: #f3f4f6; border-radius: 100px; border: 3px solid #d1d5db; margin: 15px;")
        user_photo_layout = QVBoxLayout(user_photo_frame)
        
        self.user_photo = QLabel("No User Selected")
        self.user_photo.setAlignment(Qt.AlignCenter)
        self.user_photo.setStyleSheet("font-size: 16px; color: #6b7280; font-family: 'Segoe UI', sans-serif;")
        
        user_photo_layout.addWidget(self.user_photo)
        
        name_label = QLabel("Name:")
        name_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.user_name = QLabel("Select a user from the list")
        self.user_name.setStyleSheet("font-size: 14px; color: #6b7280; font-family: 'Segoe UI', sans-serif;")
        
        attendance_label = QLabel("Attendance Records:")
        attendance_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.user_attendance = QLabel("N/A")
        self.user_attendance.setStyleSheet("font-size: 14px; color: #6b7280; font-family: 'Segoe UI', sans-serif;")
        
        last_seen_label = QLabel("Last Seen:")
        last_seen_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.user_last_seen = QLabel("N/A")
        self.user_last_seen.setStyleSheet("font-size: 14px; color: #6b7280; font-family: 'Segoe UI', sans-serif;")
        
        action_layout = QHBoxLayout()
        self.remove_user_button = QPushButton("Remove User")
        self.remove_user_button.setIcon(QIcon(REMOVE_USER_ICON))
        self.remove_user_button.setEnabled(False)
        self.remove_user_button.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #dc2626, stop:1 #b91c1c); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-family: 'Segoe UI', sans-serif; font-weight: 600; transition: background 0.3s; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b91c1c, stop:1 #991b1b); } QPushButton:disabled { background: #f3f4f6; color: #6b7280; }")
        
        action_layout.addStretch()
        action_layout.addWidget(self.remove_user_button)
        
        detail_content = QVBoxLayout()
        detail_content.setAlignment(Qt.AlignCenter)
        detail_content.setSpacing(10)
        detail_content.addWidget(user_photo_frame, 0, Qt.AlignCenter)
        detail_content.addWidget(name_label)
        detail_content.addWidget(self.user_name)
        detail_content.addWidget(attendance_label)
        detail_content.addWidget(self.user_attendance)
        detail_content.addWidget(last_seen_label)
        detail_content.addWidget(self.user_last_seen)
        detail_content.addStretch()
        
        user_detail_layout.addWidget(detail_header)
        user_detail_layout.addLayout(detail_content)
        user_detail_layout.addLayout(action_layout)
        
        content_layout.addWidget(user_list_frame, 1)
        content_layout.addWidget(user_detail_frame, 2)
        
        layout.addLayout(header_layout)
        layout.addLayout(content_layout, 1)
        
        self.add_user_button.clicked.connect(self.add_new_user)
        self.remove_user_button.clicked.connect(self.remove_selected_user)
        self.user_list.itemSelectionChanged.connect(self.update_user_details)
        self.load_user_list()
        print("Users tab setup complete")

    def setup_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        title = QLabel("System Settings")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        layout.addWidget(title)
        
        settings_frame = QFrame()
        settings_frame.setFrameShape(QFrame.StyledPanel)
        settings_frame.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f9fafb); border-radius: 12px; border: 1px solid #e5e7eb; }")
        shadow = QGraphicsDropShadowEffect(settings_frame)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 50))
        settings_frame.setGraphicsEffect(shadow)
        
        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(20, 20, 20, 20)
        
        recognition_group = QGroupBox("Recognition Settings")
        recognition_group.setStyleSheet("QGroupBox { font-size: 18px; font-weight: 600; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 20px; padding-top: 20px; font-family: 'Segoe UI', sans-serif; color: #1f2937; } QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }")
        recognition_layout = QGridLayout(recognition_group)
        recognition_layout.setColumnStretch(1, 1)
        recognition_layout.setColumnStretch(2, 2)
        recognition_layout.setSpacing(15)
        
        tolerance_label = QLabel("Face Recognition Tolerance:")
        tolerance_label.setStyleSheet("font-size: 14px; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.tolerance_input = QLineEdit(str(TOLERANCE))  # Fixed typo: TgrimANCE to TOLERANCE
        self.tolerance_input.setStyleSheet("QLineEdit { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #374151; } QLineEdit:focus { border: 2px solid #3b82f6; background-color: #f0f7ff; }")
        tolerance_help = QLabel("Lower values are more strict (0.4-0.6 recommended)")
        tolerance_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic; font-family: 'Segoe UI', sans-serif;")
        
        delay_label = QLabel("Recognition Delay (seconds):")
        delay_label.setStyleSheet("font-size: 14px; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(1, 86400)
        self.delay_spinbox.setValue(RECOGNITION_DELAY)
        self.delay_spinbox.setStyleSheet("QSpinBox { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #374151; } QSpinBox::up-button, QSpinBox::down-button { width: 20px; background: #e5e7eb; border-radius: 4px; } QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #d1d5db; }")
        delay_help = QLabel("Time between recording the same person (max 24 hours)")
        delay_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic; font-family: 'Segoe UI', sans-serif;")
        
        model_label = QLabel("Recognition Model:")
        model_label.setStyleSheet("font-size: 14px; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["hog", "cnn"])
        self.model_combo.setCurrentText(MODEL)
        self.model_combo.setStyleSheet("QComboBox { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #374151; } QComboBox:hover { border: 1px solid #3b82f6; } QComboBox::drop-down { border: none; width: 30px; } QComboBox::down-arrow { image: url(icons/down_arrow.png); width: 12px; height: 12px; }")
        model_help = QLabel("HOG is faster, CNN is more accurate but slower")
        model_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic; font-family: 'Segoe UI', sans-serif;")
        
        recognition_layout.addWidget(tolerance_label, 0, 0)
        recognition_layout.addWidget(self.tolerance_input, 0, 1)
        recognition_layout.addWidget(tolerance_help, 0, 2)
        recognition_layout.addWidget(delay_label, 1, 0)
        recognition_layout.addWidget(self.delay_spinbox, 1, 1)
        recognition_layout.addWidget(delay_help, 1, 2)
        recognition_layout.addWidget(model_label, 2, 0)
        recognition_layout.addWidget(self.model_combo, 2, 1)
        recognition_layout.addWidget(model_help, 2, 2)
        
        greeting_group = QGroupBox("Greeting Settings")
        greeting_group.setStyleSheet("QGroupBox { font-size: 18px; font-weight: 600; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 20px; padding-top: 20px; font-family: 'Segoe UI', sans-serif; color: #1f2937; } QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }")
        greeting_layout = QGridLayout(greeting_group)
        greeting_layout.setSpacing(15)

        greeting_label = QLabel("Enable Greetings:")
        greeting_label.setStyleSheet("font-size: 14px; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.greeting_checkbox = QCheckBox()
        self.greeting_checkbox.setChecked(True)
        self.greeting_checkbox.setStyleSheet("margin-left: 10px;")
        greeting_help = QLabel("Show welcome message when a user is recognized")
        greeting_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic; font-family: 'Segoe UI', sans-serif;")

        template_label = QLabel("Greeting Template:")
        template_label.setStyleSheet("font-size: 14px; color: #1f2937; font-family: 'Segoe UI', sans-serif;")
        self.greeting_template = QLineEdit("Welcome back, {name}! It's {time} on {date}")
        self.greeting_template.setStyleSheet("QLineEdit { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #374151; } QLineEdit:focus { border: 2px solid #3b82f6; background-color: #f0f7ff; }")
        template_help = QLabel("Use {name}, {time}, {date} as placeholders")
        template_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic; font-family: 'Segoe UI', sans-serif;")

        preview_btn = QPushButton("Preview Greeting")
        preview_btn.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 12pt; font-family: 'Segoe UI', sans-serif; font-weight: 600; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #1d4ed8); }")
        preview_btn.clicked.connect(self.preview_greeting)

        greeting_layout.addWidget(greeting_label, 0, 0)
        greeting_layout.addWidget(self.greeting_checkbox, 0, 1)
        greeting_layout.addWidget(greeting_help, 0, 2)
        greeting_layout.addWidget(template_label, 1, 0)
        greeting_layout.addWidget(self.greeting_template, 1, 1)
        greeting_layout.addWidget(template_help, 1, 2)
        greeting_layout.addWidget(preview_btn, 2, 1)
        
        settings_layout.addWidget(recognition_group)
        settings_layout.addWidget(greeting_group)
        settings_layout.addStretch()
        layout.addWidget(settings_frame)
        
        self.tolerance_input.textChanged.connect(self.update_tolerance)
        self.delay_spinbox.valueChanged.connect(self.update_delay)
        self.model_combo.currentTextChanged.connect(self.update_model)
        print("Settings tab setup complete")

    def load_stylesheet(self):
        return "QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f3f4f6, stop:1 #e5e7eb); } QWidget { font-family: 'Segoe UI', sans-serif; }"

    def load_known_faces(self):
        global known_face_encodings, known_face_names
        known_face_encodings = []
        known_face_names = []
        
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
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        known_face_encodings.append(encoding[0])
                        known_face_names.append(name)
                        print(f"Loaded '{name}' from '{image_path}'")
                    else:
                        print(f"No face detected in '{image_path}'")
                except Exception as e:
                    print(f"Error loading '{image_path}': {str(e)}")
        print(f"Total known faces loaded: {len(known_face_names)}")
        self.status_faces.set_data("Known Users", str(len(known_face_names)), "#3b82f6")

    def load_attendance_records(self):
        self.attendance_model.setRowCount(0)
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    for row in reader:
                        if len(row) == 3:
                            name, date, time = row
                            row_count = self.attendance_model.rowCount()
                            self.attendance_model.insertRow(row_count)
                            self.attendance_model.setItem(row_count, 0, QStandardItem(name))
                            self.attendance_model.setItem(row_count, 1, QStandardItem(time))
                            self.attendance_model.setItem(row_count, 2, QStandardItem(date))
            print(f"Loaded {self.attendance_model.rowCount()} attendance records")
        self.update_today_count()
        self.update_trends()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            self.status_webcam.set_data("Webcam Status", "Disconnected", "#dc2626")
            self.video_label.setText("Webcam feed unavailable")
            return
        
        self.status_webcam.set_data("Webcam Status", "Active", "#16a34a")
        
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
        if not face_locations:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return

        clustering = DBSCAN(eps=50, min_samples=1).fit(face_locations)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        current_time = time.time()

        for label in set(clustering.labels_):
            if label == -1:
                continue
            cluster_indices = [i for i, lbl in enumerate(clustering.labels_) if lbl == label]
            cluster_locations = [face_locations[i] for i in cluster_indices]
            cluster_encodings = [face_encodings[i] for i in cluster_indices]

            for (top, right, bottom, left), face_encoding in zip(cluster_locations, cluster_encodings):
                top = int(top / RESIZE_FACTOR)
                right = int(right / RESIZE_FACTOR)
                bottom = int(bottom / RESIZE_FACTOR)  # Fixed typo: should be 'bottom'
                left = int(left / RESIZE_FACTOR)
                
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    
                    if self.greeting_checkbox.isChecked():
                        greeting_message = self.greeting_template.text().format(
                            name=name,
                            time=datetime.now().strftime('%H:%M:%S'),
                            date=datetime.now().strftime('%Y-%m-%d')
                        )
                        self.greeting_label.setText(greeting_message)
                        self.greeting_label.setVisible(True)
                        self.greeting_timer.start(3000)
                    
                    last_time = self.last_recognition_time_ui.get(name, 0)
                    if current_time - last_time >= RECOGNITION_DELAY:
                        print(f"Detected and recognized '{name}' (first time or after delay)")
                        self.mark_attendance(name)
                        self.last_recognition_time_ui[name] = current_time
                        # New: Update activity log
                        self.activity_list.addItem(f"{name} detected at {datetime.now().strftime('%H:%M:%S')}")
                        if self.activity_list.count() > 10:  # Keep only last 10 entries
                            self.activity_list.takeItem(0)
                else:
                    print(f"Detected an unknown face in cluster {label}")
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), FONT_THICKNESS)
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def hide_greeting(self):
        self.greeting_label.setVisible(False)

    def has_attendance_today(self, name, date_string):
        for row in range(self.attendance_model.rowCount()):
            row_name = self.attendance_model.item(row, 0).text()
            row_date = self.attendance_model.item(row, 2).text()
            if row_name == name and row_date == date_string:
                return True
        return False

    def mark_attendance(self, name):
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        
        if self.has_attendance_today(name, date_string):
            print(f"Attendance already marked for {name} on {date_string}")
            return
        
        row = self.attendance_model.rowCount()
        self.attendance_model.insertRow(row)
        self.attendance_model.setItem(row, 0, QStandardItem(name))
        self.attendance_model.setItem(row, 1, QStandardItem(time_string))
        self.attendance_model.setItem(row, 2, QStandardItem(date_string))
        
        self.update_today_count()
        self.update_trends()
        
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_string, time_string])
        print(f"Attendance marked for {name} on {date_string} at {time_string}")

    def update_today_count(self):
        today = datetime.now().strftime('%Y-%m-%d')
        present_count = sum(1 for row in range(self.attendance_model.rowCount()) 
                            if self.attendance_model.item(row, 2).text() == today)
        self.status_today.set_data("Today's Attendance", str(present_count), "#16a34a")

    def update_trends(self):
        self.trend_series.clear()
        attendance_by_date = {}
        today = datetime.now().date()
        for i in range(7):  # Last 7 days
            date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            attendance_by_date[date] = 0
        
        for row in range(self.attendance_model.rowCount()):
            date = self.attendance_model.item(row, 2).text()
            if date in attendance_by_date:
                attendance_by_date[date] += 1
        
        for date, count in sorted(attendance_by_date.items()):
            self.trend_series.append(float((today - datetime.strptime(date, '%Y-%m-%d').date()).days), count)
        
        self.trend_chart.removeAxis(self.trend_chart.axisX())
        self.trend_chart.removeAxis(self.trend_chart.axisY())
        self.trend_chart.createDefaultAxes()
        self.trend_chart.axisX().setTitleText("Days Ago")
        self.trend_chart.axisY().setTitleText("Attendance Count")

    def preview_greeting(self):
        template = self.greeting_template.text()
        try:
            preview = template.format(
                name="Test User",
                time=datetime.now().strftime('%H:%M:%S'),
                date=datetime.now().strftime('%Y-%m-%d')
            )
            QMessageBox.information(self, "Greeting Preview", preview)
        except KeyError as e:
            QMessageBox.warning(self, "Error", f"Invalid placeholder: {str(e)}")

    def export_data(self):
        options = QFileDialog.Options()
        file_name, selected_filter = QFileDialog.getSaveFileName(self, "Export Attendance Data", "", "PDF Files (*.pdf);;CSV Files (*.csv)", options=options)
        if file_name:
            try:
                if selected_filter == "PDF Files (*.pdf)":
                    if not file_name.endswith('.pdf'):
                        file_name += '.pdf'
                    doc = SimpleDocTemplate(file_name, pagesize=letter)
                    elements = []
                    row_count = self.attendance_model.rowCount()
                    data = [["Name", "Time", "Date"]] + [
                        [self.attendance_model.item(r, c).text() for c in range(3)] 
                        for r in range(row_count)
                    ]
                    elements.append(Table(data))
                    
                    attendance_by_date = {}
                    for row in range(row_count):
                        date = self.attendance_model.item(row, 2).text()
                        attendance_by_date[date] = attendance_by_date.get(date, 0) + 1
                    
                    drawing = Drawing(400, 200)
                    bc = VerticalBarChart()
                    bc.x = 50
                    bc.y = 50
                    bc.height = 125
                    bc.width = 300
                    bc.data = [list(attendance_by_date.values())]
                    bc.categoryAxis.categoryNames = list(attendance_by_date.keys())
                    bc.categoryAxis.labels.angle = 45
                    bc.categoryAxis.labels.boxAnchor = 'ne'
                    bc.valueAxis.labelTextFormat = '%d'
                    drawing.add(bc)
                    elements.append(drawing)
                    
                    doc.build(elements)
                    print(f"Data exported to PDF: {file_name}")
                    QMessageBox.information(self, "Success", "Attendance data exported successfully as PDF with graph.")
                else:  # CSV export
                    if not file_name.endswith('.csv'):
                        file_name += '.csv'
                    with open(file_name, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Name", "Time", "Date"])
                        for row in range(self.attendance_model.rowCount()):
                            writer.writerow([self.attendance_model.item(row, i).text() for i in range(3)])
                    print(f"Data exported to CSV: {file_name}")
                    QMessageBox.information(self, "Success", "Attendance data exported successfully as CSV.")
            except Exception as e:
                print(f"Export failed: {str(e)}")
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def load_user_list(self):
        self.user_list.clear()
        for name in known_face_names:
            self.user_list.addItem(name)
        print(f"User list updated with {len(known_face_names)} users")

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
                image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                try:
                    pixmap = QPixmap(dialog.image_path)
                    if pixmap.save(image_path, "JPG"):
                        print(f"Saved new user image to '{image_path}'")
                        self.load_known_faces()
                        self.load_user_list()
                        self.status_faces.set_data("Known Users", str(len(known_face_names)), "#3b82f6")
                    else:
                        print(f"Failed to save image to '{image_path}'")
                        QMessageBox.critical(self, "Error", "Failed to save the image.")
                except Exception as e:
                    print(f"Error saving image: {str(e)}")
                    QMessageBox.critical(self, "Error", f"Error saving image: {str(e)}")
            else:
                print("No image selected, aborting")
                QMessageBox.warning(self, "Error", "Please select an image.")
        else:
            print("Dialog cancelled")

    def remove_selected_user(self):
        current_item = self.user_list.currentItem()
        if not current_item:
            print("No user selected for removal")
            QMessageBox.warning(self, "Error", "Please select a user to remove.")
            return

        name = current_item.text()
        image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed user '{name}' from '{image_path}'")
                
                if name in known_face_names:
                    index = known_face_names.index(name)
                    known_face_names.pop(index)
                    known_face_encodings.pop(index)
                    print(f"Removed '{name}' from known faces list")
                
                self.user_list.clearSelection()
                self.load_user_list()
                self.status_faces.set_data("Known Users", str(len(known_face_names)), "#3b82f6")
                self.update_user_details()
                QMessageBox.information(self, "Success", f"User '{name}' has been removed successfully.")
            else:
                print(f"Image file '{image_path}' not found")
                QMessageBox.warning(self, "Error", f"User image for '{name}' not found.")
        except Exception as e:
            print(f"Error removing user '{name}': {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to remove user '{name}': {str(e)}")

    def update_user_details(self):
        current_item = self.user_list.currentItem()
        if current_item:
            name = current_item.text()
            self.user_name.setText(name)
            self.remove_user_button.setEnabled(True)
            
            image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                self.user_photo.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.user_photo.setText("Image Not Found")
            
            attendance_count = sum(1 for row in range(self.attendance_model.rowCount())
                                 if self.attendance_model.item(row, 0).text() == name)
            self.user_attendance.setText(f"{attendance_count} records")
            
            last_seen = "Never"
            for row in range(self.attendance_model.rowCount()):
                if self.attendance_model.item(row, 0).text() == name:
                    last_seen = f"{self.attendance_model.item(row, 2).text()} {self.attendance_model.item(row, 1).text()}"
            self.user_last_seen.setText(last_seen)
        else:
            self.user_name.setText("Select a user from the list")
            self.user_photo.setText("No User Selected")
            self.user_attendance.setText("N/A")
            self.user_last_seen.setText("N/A")
            self.remove_user_button.setEnabled(False)

    def update_tolerance(self, value):
        global TOLERANCE
        try:
            TOLERANCE = float(value)
            print(f"Tolerance updated to {TOLERANCE}")
        except ValueError:
            print("Invalid tolerance value, keeping previous value")

    def update_delay(self, value):
        global RECOGNITION_DELAY
        RECOGNITION_DELAY = value
        print(f"Recognition delay updated to {RECOGNITION_DELAY} seconds")

    def update_model(self, value):
        global MODEL
        MODEL = value
        self.status_recognition.set_data("Recognition Model", MODEL, "#8b5cf6")
        print(f"Recognition model updated to {MODEL}")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created directory: {KNOWN_FACES_DIR}")
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])
        print(f"Created attendance file: {ATTENDANCE_FILE}")
    
    def load_known_faces():
        global known_face_encodings, known_face_names
        known_face_encodings = []
        known_face_names = []
        
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        known_face_encodings.append(encoding[0])
                        known_face_names.append(name)
                        print(f"Loaded '{name}' from '{image_path}'")
                    else:
                        print(f"No face detected in '{image_path}'")
                except Exception as e:
                    print(f"Error loading '{image_path}': {str(e)}")
    
    load_known_faces()
    
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())