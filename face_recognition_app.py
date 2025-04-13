import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import csv
import cv2
import face_recognition
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.cluster import DBSCAN
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QListWidget, QPushButton, QTabWidget, QTableView,
                             QHeaderView, QFrame, QComboBox, QLineEdit, QSpinBox,
                             QGroupBox, QCheckBox, QFileDialog, QMessageBox,
                             QGraphicsDropShadowEffect, QSizePolicy, QGridLayout, QDialog)
from PyQt5.QtGui import (QImage, QPixmap, QIcon, QStandardItemModel, QStandardItem,
                         QColor)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from login_dialog import LoginDialog
from add_user_dialog import AddUserDialog
from student_registration_dialog import StudentRegistrationDialog
from status_widget import StatusWidget

# Configuration
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_FILE = 'attendance.csv'
STUDENTS_FILE = 'students.csv'
TOLERANCE = 0.5
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
MODEL = 'hog'
RESIZE_FACTOR = 0.25
RECOGNITION_DELAY = 10  # 24 hours in seconds

# Email Configuration (Replace with your credentials)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "hotelardilla@gmail.com"  # Replace with your email
SENDER_PASSWORD = "qviw qyaj kwsj hfgd"  # Replace with your app-specific password

# Icon Paths (Ensure these files exist in an 'icons' directory)
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

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Attendance System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setWindowIcon(QIcon("icons/app_icon.png"))
        self.setStyleSheet("QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f3f4f6, stop:1 #e5e7eb); } QWidget { font-family: 'Segoe UI', sans-serif; }")
        
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
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: #f3f4f6; }
            QTabBar::tab { padding: 12px 24px; margin: 0 4px; font-size: 14px; font-family: 'Segoe UI', sans-serif; font-weight: 600; min-width: 120px; border-top-left-radius: 8px; border-top-right-radius: 8px; background: #e5e7eb; color: #4b5563; }
            QTabBar::tab:selected { background: #3b82f6; color: white; }
            QTabBar::tab:hover:!selected { background: #f3f4f6; color: #1f2937; }
        """)
        
        self.dashboard_tab = QWidget()
        self.attendance_tab = QWidget()
        self.users_tab = QWidget()
        self.settings_tab = QWidget()
        
        self.tabs.addTab(self.dashboard_tab, QIcon(DASHBOARD_ICON), "Dashboard")
        self.tabs.addTab(self.attendance_tab, QIcon(ATTENDANCE_ICON), "Attendance")
        self.tabs.addTab(self.users_tab, QIcon(USERS_ICON), "Users")
        self.tabs.addTab(self.settings_tab, QIcon(SETTINGS_ICON), "Settings")
        
        self.main_layout.addWidget(self.tabs)
        
        # Initialize StatusWidgets
        self.status_webcam = StatusWidget()
        self.status_webcam.set_data("Webcam Status", "Connecting...", "#dc2626")
        self.status_faces = StatusWidget()
        self.status_faces.set_data("Known Users", "0", "#3b82f6")
        self.status_today = StatusWidget()
        self.status_today.set_data("Today's Attendance", "0", "#16a34a")
        self.status_recognition = StatusWidget()
        self.status_recognition.set_data("Recognition Model", MODEL, "#8b5cf6")
        
        # Initialize Trend Chart components
        self.trend_chart = QChart()
        self.trend_chart.setTitle("Attendance Over Time")
        self.trend_series = QLineSeries()
        self.trend_chart.addSeries(self.trend_series)
        self.trend_chart.createDefaultAxes()
        self.trend_view = QChartView(self.trend_chart)
        self.trend_view.setMinimumHeight(200)
        
        self.last_recognition_time_ui = {}
        self.load_known_faces()
        self.load_attendance_records()
        
        self.setup_dashboard_tab()
        self.setup_attendance_tab()
        self.setup_users_tab()
        self.setup_settings_tab()
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open webcam.")
            self.cap = None
        else:
            self.status_webcam.set_data("Webcam Status", "Connected", "#16a34a")
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        if self.cap:
            self.timer.start(30)
        
        self.greeting_timer = QTimer(self)
        self.greeting_timer.setSingleShot(True)
        self.greeting_timer.timeout.connect(self.hide_greeting)

    def setup_dashboard_tab(self):
        layout = QVBoxLayout(self.dashboard_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        title_layout = QVBoxLayout()
        title = QLabel("Attendance Dashboard")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1f2937;")
        subtitle = QLabel("Real-time facial recognition attendance monitoring")
        subtitle.setStyleSheet("font-size: 16px; color: #6b7280;")
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
        video_header.setStyleSheet("font-size: 18px; font-weight: 600; color: #1f2937;")
        video_header.setAlignment(Qt.AlignCenter)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #e5e7eb; border-radius: 10px; background-color: #f9fafb;")
        
        self.greeting_label = QLabel("")
        self.greeting_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #16a34a; background: #f0fdf4; padding: 10px; border-radius: 8px; border: 1px solid #bbf7d0;")
        self.greeting_label.setAlignment(Qt.AlignCenter)
        self.greeting_label.setVisible(False)
        
        video_footer = QLabel("System actively monitoring for registered users")
        video_footer.setStyleSheet("font-size: 13px; color: #6b7280;")
        video_footer.setAlignment(Qt.AlignCenter)
        
        activity_frame = QFrame()
        activity_frame.setFrameShape(QFrame.StyledPanel)
        activity_frame.setStyleSheet("QFrame { background: #ffffff; border-radius: 10px; border: 1px solid #e5e7eb; }")
        activity_layout = QVBoxLayout(activity_frame)
        activity_header = QLabel("Recent Activity Log")
        activity_header.setStyleSheet("font-size: 14px; font-weight: 600; color: #1f2937;")
        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet("QListWidget { border: none; background: transparent; color: #374151; } QListWidget::item { padding: 8px; font-size: 13px; border-bottom: 1px solid #f3f4f6; }")
        self.activity_list.setMaximumHeight(150)
        activity_layout.addWidget(activity_header)
        activity_layout.addWidget(self.activity_list)
        
        video_layout.addWidget(video_header)
        video_layout.addWidget(self.video_label, 1)
        video_layout.addWidget(self.greeting_label)
        video_layout.addWidget(video_footer)
        video_layout.addWidget(activity_frame)
        
        status_panel = QVBoxLayout()
        status_panel.setSpacing(20)
        
        status_row1 = QHBoxLayout()
        status_row1.addWidget(self.status_webcam)
        status_row1.addWidget(self.status_faces)
        
        status_row2 = QHBoxLayout()
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
        trend_header.setStyleSheet("font-size: 16px; font-weight: 600; color: #1f2937;")
        trend_layout.addWidget(trend_header)
        trend_layout.addWidget(self.trend_view)
        
        status_panel.addLayout(status_row1)
        status_panel.addLayout(status_row2)
        status_panel.addWidget(trend_frame, 1)
        
        main_content.addWidget(video_panel, 3)
        main_content.addLayout(status_panel, 2)
        
        layout.addLayout(main_content, 1)
        self.update_trends()

    def setup_attendance_tab(self):
        layout = QVBoxLayout(self.attendance_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        header_layout = QHBoxLayout()
        title = QLabel("Attendance Records")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1f2937;")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        export_btn = QPushButton("Export Data")
        export_btn.setIcon(QIcon(EXPORT_ICON))
        export_btn.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #16a34a, stop:1 #15803d); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 600; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #15803d, stop:1 #166534); }")
        export_btn.clicked.connect(self.export_data)
        
        date_filter = QComboBox()
        date_filter.addItems(["All Dates", "Today", "Yesterday", "This Week", "This Month"])
        date_filter.setStyleSheet("QComboBox { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; min-width: 150px; font-size: 13px; color: #374151; } QComboBox:hover { border: 1px solid #3b82f6; }")
        
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
        self.attendance_table.setStyleSheet("QTableView { border: none; background: transparent; color: #374151; selection-background-color: #eff6ff; } QHeaderView::section { background: #f9fafb; padding: 12px; border: none; border-bottom: 1px solid #e5e7eb; font-weight: 600; font-size: 14px; color: #1f2937; } QTableView::item { padding: 12px; border-bottom: 1px solid #f3f4f6; font-size: 13px; }")
        
        table_layout.addWidget(self.attendance_table)
        
        layout.addLayout(header_layout)
        layout.addWidget(table_frame, 1)

    def setup_users_tab(self):
        layout = QVBoxLayout(self.users_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        header_layout = QHBoxLayout()
        title = QLabel("User Management")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1f2937;")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.add_user_button = QPushButton("Add New User")
        self.add_user_button.setIcon(QIcon(ADD_USER_ICON))
        self.add_user_button.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 600; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #1d4ed8); }")
        self.register_student_btn = QPushButton("Register Student")
        self.register_student_btn.setIcon(QIcon(ADD_USER_ICON))
        self.register_student_btn.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10b981, stop:1 #059669); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 600; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #059669, stop:1 #047857); }")
        
        header_layout.addWidget(title)
        header_layout.addWidget(spacer)
        header_layout.addWidget(self.register_student_btn)
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
        user_search.setStyleSheet("QLineEdit { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #f9fafb; font-size: 13px; color: #374151; margin-bottom: 15px; } QLineEdit:focus { border: 2px solid #3b82f6; }")
        
        self.user_list = QListWidget()
        self.user_list.setStyleSheet("QListWidget { border: none; background: transparent; color: #374151; } QListWidget::item { border-bottom: 1px solid #f3f4f6; padding: 14px; font-size: 14px; } QListWidget::item:selected { background: #eff6ff; color: #1f2937; border-radius: 6px; }")
        
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
        detail_header.setStyleSheet("font-size: 20px; font-weight: 600; color: #1f2937;")
        
        user_photo_frame = QFrame()
        user_photo_frame.setFixedSize(200, 200)
        user_photo_frame.setStyleSheet("background-color: #f3f4f6; border-radius: 100px; border: 3px solid #d1d5db; margin: 15px;")
        user_photo_layout = QVBoxLayout(user_photo_frame)
        
        self.user_photo = QLabel("No User Selected")
        self.user_photo.setAlignment(Qt.AlignCenter)
        self.user_photo.setStyleSheet("font-size: 16px; color: #6b7280;")
        
        user_photo_layout.addWidget(self.user_photo)
        
        name_label = QLabel("Name:")
        name_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2937;")
        self.user_name = QLabel("Select a user from the list")
        self.user_name.setStyleSheet("font-size: 14px; color: #6b7280;")
        
        attendance_label = QLabel("Attendance Records:")
        attendance_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2937;")
        self.user_attendance = QLabel("N/A")
        self.user_attendance.setStyleSheet("font-size: 14px; color: #6b7280;")
        
        last_seen_label = QLabel("Last Seen:")
        last_seen_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #1f2937;")
        self.user_last_seen = QLabel("N/A")
        self.user_last_seen.setStyleSheet("font-size: 14px; color: #6b7280;")
        
        action_layout = QHBoxLayout()
        self.remove_user_button = QPushButton("Remove User")
        self.remove_user_button.setIcon(QIcon(REMOVE_USER_ICON))
        self.remove_user_button.setEnabled(False)
        self.remove_user_button.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #dc2626, stop:1 #b91c1c); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 600; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b91c1c, stop:1 #991b1b); } QPushButton:disabled { background: #f3f4f6; color: #6b7280; }")
        
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
        self.register_student_btn.clicked.connect(self.register_student)
        self.user_list.itemSelectionChanged.connect(self.update_user_details)
        self.load_user_list()

    def setup_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        title = QLabel("System Settings")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #1f2937;")
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
        recognition_group.setStyleSheet("QGroupBox { font-size: 18px; font-weight: 600; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 20px; padding-top: 20px; color: #1f2937; } QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }")
        recognition_layout = QGridLayout(recognition_group)
        recognition_layout.setColumnStretch(1, 1)
        recognition_layout.setColumnStretch(2, 2)
        recognition_layout.setSpacing(15)
        
        tolerance_label = QLabel("Face Recognition Tolerance:")
        tolerance_label.setStyleSheet("font-size: 14px; color: #1f2937;")
        self.tolerance_input = QLineEdit(str(TOLERANCE))
        self.tolerance_input.setStyleSheet("QLineEdit { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-size: 13px; color: #374151; } QLineEdit:focus { border: 2px solid #3b82f6; }")
        tolerance_help = QLabel("Lower values are more strict (0.4-0.6 recommended)")
        tolerance_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic;")
        
        delay_label = QLabel("Recognition Delay (seconds):")
        delay_label.setStyleSheet("font-size: 14px; color: #1f2937;")
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(1, 86400)
        self.delay_spinbox.setValue(RECOGNITION_DELAY)
        self.delay_spinbox.setStyleSheet("QSpinBox { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-size: 13px; color: #374151; } QSpinBox::up-button, QSpinBox::down-button { width: 20px; background: #e5e7eb; }")
        delay_help = QLabel("Time between recording the same person (max 24 hours)")
        delay_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic;")
        
        model_label = QLabel("Recognition Model:")
        model_label.setStyleSheet("font-size: 14px; color: #1f2937;")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["hog", "cnn"])
        self.model_combo.setCurrentText(MODEL)
        self.model_combo.setStyleSheet("QComboBox { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-size: 13px; color: #374151; } QComboBox:hover { border: 1px solid #3b82f6; }")
        model_help = QLabel("HOG is faster, CNN is more accurate but slower")
        model_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic;")
        
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
        greeting_group.setStyleSheet("QGroupBox { font-size: 18px; font-weight: 600; border: 1px solid #e5e7eb; border-radius: 8px; margin-top: 20px; padding-top: 20px; color: #1f2937; } QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; }")
        greeting_layout = QGridLayout(greeting_group)
        greeting_layout.setSpacing(15)

        greeting_label = QLabel("Enable Greetings:")
        greeting_label.setStyleSheet("font-size: 14px; color: #1f2937;")
        self.greeting_checkbox = QCheckBox()
        self.greeting_checkbox.setChecked(True)
        self.greeting_checkbox.setStyleSheet("margin-left: 10px;")
        greeting_help = QLabel("Show welcome message when a user is recognized")
        greeting_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic;")

        template_label = QLabel("Greeting Template:")
        template_label.setStyleSheet("font-size: 14px; color: #1f2937;")
        self.greeting_template = QLineEdit("Welcome back, {name}! It's {time} on {date}")
        self.greeting_template.setStyleSheet("QLineEdit { padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; background-color: #ffffff; font-size: 13px; color: #374151; } QLineEdit:focus { border: 2px solid #3b82f6; }")
        template_help = QLabel("Use {name}, {time}, {date} as placeholders")
        template_help.setStyleSheet("font-size: 12px; color: #6b7280; font-style: italic;")

        preview_btn = QPushButton("Preview Greeting")
        preview_btn.setStyleSheet("QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b82f6, stop:1 #2563eb); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 12pt; font-weight: 600; } QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #1d4ed8); }")
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

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
        
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        self.known_face_encodings.append(encoding[0])
                        self.known_face_names.append(name)
                    else:
                        self.activity_list.addItem(f"Warning: No face detected in {filename}")
                except Exception as e:
                    self.activity_list.addItem(f"Error loading {filename}: {str(e)}")
                    continue
        self.status_faces.set_data("Known Users", str(len(self.known_face_names)), "#3b82f6")

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
        self.update_today_count()
        self.update_trends()

    def update_frame(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.status_webcam.set_data("Webcam Status", "Disconnected", "#dc2626")
                    self.video_label.setText("Webcam feed unavailable")
                    return
                else:
                    self.status_webcam.set_data("Webcam Status", "Connected", "#16a34a")
            
            ret, frame = self.cap.read()
            if not ret:
                self.status_webcam.set_data("Webcam Status", "Disconnected", "#dc2626")
                self.video_label.setText("Webcam feed unavailable")
                return
            
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
            self.activity_list.addItem(f"Detected {len(face_locations)} faces in frame")
            if not face_locations:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                if qt_image.isNull():
                    self.activity_list.addItem("Error: Failed to create QImage from frame")
                    return
                pixmap = QPixmap.fromImage(qt_image)
                if pixmap.isNull():
                    self.activity_list.addItem("Error: Failed to create QPixmap from QImage")
                    return
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
                    bottom = int(bottom / RESIZE_FACTOR)
                    left = int(left / RESIZE_FACTOR)
                    
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=TOLERANCE)
                    name = "Unknown"
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                        
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
                        if current_time - last_time >= self.delay_spinbox.value():
                            self.mark_attendance(name)
                            self.last_recognition_time_ui[name] = current_time
                            self.activity_list.addItem(f"{name} detected at {datetime.now().strftime('%H:%M:%S')}")
                            if self.activity_list.count() > 10:
                                self.activity_list.takeItem(0)
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), FONT_THICKNESS)
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            if qt_image.isNull():
                self.activity_list.addItem("Error: Failed to create QImage from frame")
                return
            pixmap = QPixmap.fromImage(qt_image)
            if pixmap.isNull():
                self.activity_list.addItem("Error: Failed to create QPixmap from QImage")
                return
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.activity_list.addItem(f"Error in update_frame: {str(e)}")
            self.status_webcam.set_data("Webcam Status", "Error", "#dc2626")
        
    def hide_greeting(self):
        self.greeting_label.setVisible(False)

    def has_attendance_today(self, name, date_string):
        for row in range(self.attendance_model.rowCount()):
            row_name = self.attendance_model.item(row, 0).text()
            row_date = self.attendance_model.item(row, 2).text()
            if row_name == name and row_date == date_string:
                return True
        return False

    def send_attendance_email(self, name, email, timestamp):
        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = email
            msg['Subject'] = "Attendance Recorded"

            body = f"Dear {name},\n\nYour attendance was recorded at {timestamp}.\n\nBest regards,\nIntelligent Attendance System"
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, msg.as_string())
            server.quit()
            self.activity_list.addItem(f"Email sent to {name} at {email}")
        except Exception as e:
            self.activity_list.addItem(f"Failed to send email to {name}: {str(e)}")

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
            self.activity_list.addItem(f"Email sent to {to_email}")
        except Exception as e:
            self.activity_list.addItem(f"Failed to send email: {str(e)}")

    def capture_face(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.activity_list.addItem("Error: Could not open webcam for face capture.")
                return []

        images = []
        self.activity_list.addItem("Please face the camera. Capturing in 3 seconds...")
        QTimer.singleShot(3000, lambda: self.activity_list.addItem("Capturing now..."))
        time.sleep(3)  # Give user time to prepare

        for _ in range(10):  # Capture 10 frames
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(rgb_frame)
            cv2.imshow("Capturing Face", frame)
            cv2.waitKey(100)  # Wait 100ms between frames
        cv2.destroyAllWindows()
        return images

    def mark_attendance(self, name):
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        timestamp = f"{date_string} {time_string}"
        
        if self.has_attendance_today(name, date_string):
            return
        
        email = None
        if os.path.exists(STUDENTS_FILE):
            with open(STUDENTS_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['name'] == name:
                        email = row['email']
                        break
        
        row = self.attendance_model.rowCount()
        self.attendance_model.insertRow(row)
        self.attendance_model.setItem(row, 0, QStandardItem(name))
        self.attendance_model.setItem(row, 1, QStandardItem(time_string))
        self.attendance_model.setItem(row, 2, QStandardItem(date_string))
        
        self.update_today_count()
        self.update_trends()
        
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
                writer.writerow(["Name", "Date", "Time"])
            writer.writerow([name, date_string, time_string])
        
        if email:
            self.send_attendance_email(name, email, timestamp)
        else:
            self.activity_list.addItem(f"No email found for {name}")
            
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
            self.activity_list.addItem(f"Email sent to {to_email}")
        except Exception as e:
            self.activity_list.addItem(f"Failed to send email: {str(e)}")

    def update_today_count(self):
        today = datetime.now().strftime('%Y-%m-%d')
        present_count = sum(1 for row in range(self.attendance_model.rowCount()) 
                            if self.attendance_model.item(row, 2).text() == today)
        self.status_today.set_data("Today's Attendance", str(present_count), "#16a34a")

    def update_trends(self):
        self.trend_series.clear()
        attendance_by_date = {}
        today = datetime.now().date()
        for i in range(7):
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
                    QMessageBox.information(self, "Success", "Attendance data exported successfully as PDF with graph.")
                else:
                    if not file_name.endswith('.csv'):
                        file_name += '.csv'
                    with open(file_name, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Name", "Time", "Date"])
                        for row in range(self.attendance_model.rowCount()):
                            writer.writerow([self.attendance_model.item(row, i).text() for i in range(3)])
                    QMessageBox.information(self, "Success", "Attendance data exported successfully as CSV.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def load_user_list(self):
        self.user_list.clear()
        for name in self.known_face_names:
            self.user_list.addItem(name)

    def add_new_user(self):
        dialog = AddUserDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name = dialog.name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "Error", "Please enter a name.")
                return
            
            if dialog.image_path:
                final_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                try:
                    if dialog.captured_image is not None:
                        cv2.imwrite(final_image_path, dialog.captured_image)
                    else:
                        pixmap = QPixmap(dialog.image_path)
                        if not pixmap.save(final_image_path, "JPG"):
                            raise Exception("Failed to save image")
                    self.load_known_faces()
                    self.load_user_list()
                    self.status_faces.set_data("Known Users", str(len(self.known_face_names)), "#3b82f6")
                    QMessageBox.information(self, "Success", f"User '{name}' added successfully.")
                    if dialog.image_path and os.path.exists(dialog.image_path) and "_temp" in dialog.image_path:
                        os.remove(dialog.image_path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error saving image: {str(e)}")

    def remove_selected_user(self):
        selected_items = self.user_list.selectedItems()
        if not selected_items:
            return
        
        name = selected_items[0].text()
        reply = QMessageBox.question(self, "Confirm Removal", f"Are you sure you want to remove '{name}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)
            self.load_known_faces()
            self.load_user_list()
            self.status_faces.set_data("Known Users", str(len(self.known_face_names)), "#3b82f6")
            self.user_photo.setText("No User Selected")
            self.user_name.setText("Select a user from the list")
            self.user_attendance.setText("N/A")
            self.user_last_seen.setText("N/A")
            self.remove_user_button.setEnabled(False)
            QMessageBox.information(self, "Success", f"User '{name}' removed successfully.")

    def update_user_details(self):
        selected_items = self.user_list.selectedItems()
        if not selected_items:
            self.remove_user_button.setEnabled(False)
            return
        
        name = selected_items[0].text()
        self.user_name.setText(name)
        image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.user_photo.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.user_photo.setText("Image Not Found")
        
        attendance_count = sum(1 for row in range(self.attendance_model.rowCount()) 
                               if self.attendance_model.item(row, 0).text() == name)
        self.user_attendance.setText(str(attendance_count))
        
        last_seen = "N/A"
        for row in range(self.attendance_model.rowCount() - 1, -1, -1):
            if self.attendance_model.item(row, 0).text() == name:
                last_seen = f"{self.attendance_model.item(row, 2).text()} {self.attendance_model.item(row, 1).text()}"
                break
        self.user_last_seen.setText(last_seen)
        
        self.remove_user_button.setEnabled(True)

    def register_student(self):
        dialog = StudentRegistrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            result = dialog.get_result()  # Call the method with parentheses
            name = result["name"]
            student_id = result["student_id"]
            email = result["email"]
            image_path = result["image_path"]

            # Validate inputs
            if not name or not student_id:
                QMessageBox.warning(self, "Error", "Name and student ID are required.")
                return

            # Basic email validation (if email is provided)
            if email and ('@' not in email or '.' not in email):
                QMessageBox.warning(self, "Error", "Please enter a valid email address.")
                return

            # Capture face if no image was provided
            if not image_path:
                self.activity_list.addItem(f"Capturing face for {name}...")
                face_images = self.capture_face()
                if not face_images:
                    self.activity_list.addItem("Failed to capture face. Registration cancelled.")
                    return

                # Encode the captured face images to ensure at least one has a detectable face
                face_encodings = []
                selected_image = None
                for img in face_images:
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        face_encodings.append(encodings[0])
                        selected_image = img
                        break  # Use the first image with a detectable face

                if not face_encodings or not selected_image:
                    self.activity_list.addItem("No face detected during capture. Registration cancelled.")
                    return

                # Save the selected image to known_faces directory
                final_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                cv2.imwrite(final_image_path, cv2.cvtColor(selected_image, cv2.COLOR_RGB2BGR))
            else:
                # If an image was provided, copy it to known_faces directory and verify it
                final_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
                pixmap = QPixmap(image_path)
                if pixmap.isNull():
                    QMessageBox.critical(self, "Error", "Provided image is invalid or cannot be loaded.")
                    return
                if not pixmap.save(final_image_path, "JPG"):
                    QMessageBox.critical(self, "Error", "Failed to save image.")
                    return
                # Verify the image has a detectable face
                image = face_recognition.load_image_file(final_image_path)
                encodings = face_recognition.face_encodings(image)
                if not encodings:
                    self.activity_list.addItem(f"No face detected in provided image for {name}. Registration cancelled.")
                    os.remove(final_image_path)  # Clean up the invalid image
                    return

            # Save student data to students.csv
            student_data = {"name": name, "student_id": student_id, "email": email}
            file_exists = os.path.exists(STUDENTS_FILE)
            with open(STUDENTS_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["name", "student_id", "email"])
                if not file_exists or os.stat(STUDENTS_FILE).st_size == 0:
                    writer.writeheader()
                writer.writerow(student_data)

            # Reload known faces to include the new student
            self.load_known_faces()
            self.load_user_list()
            self.status_faces.set_data("Known Users", str(len(self.known_face_names)), "#3b82f6")

            self.activity_list.addItem(f"Successfully registered student: {name} (ID: {student_id})")

        # Ensure webcam is active
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.timer.start(30)
                self.status_webcam.set_data("Webcam Status", "Connected", "#16a34a")
            else:
                self.activity_list.addItem("Error: Could not reopen webcam after registration.")

    def update_tolerance(self, text):
        try:
            global TOLERANCE
            TOLERANCE = float(text)
            if not 0 <= TOLERANCE <= 1:
                raise ValueError("Tolerance must be between 0 and 1")
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid tolerance value (0-1).")
            self.tolerance_input.setText(str(TOLERANCE))

    def update_delay(self, value):
        global RECOGNITION_DELAY
        RECOGNITION_DELAY = value

    def update_model(self, text):
        global MODEL
        MODEL = text
        self.status_recognition.set_data("Recognition Model", MODEL, "#8b5cf6")

    @staticmethod
    def show_login():
        return LoginDialog()

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    login = FaceRecognitionApp.show_login()
    if login.exec_() == QDialog.Accepted:
        # Optionally retrieve credentials if needed
        username, password = login.get_credentials()
        window = FaceRecognitionApp()
        window.show()
        sys.exit(app.exec_())