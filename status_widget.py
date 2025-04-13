from PyQt5.QtWidgets import QFrame, QVBoxLayout, QGraphicsDropShadowEffect , QLabel
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

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