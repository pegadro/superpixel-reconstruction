from PyQt5.QtWidgets import QWidget, QHBoxLayout, QScrollArea, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

class ImageCarousel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Initialize the UI components for the image carousel
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.widget = QWidget()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)
        self.inner_layout = QHBoxLayout(self.widget)
        self.inner_layout.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.scroll)

        self.setFixedHeight(120)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def add_image(self, pixmap):
        # Add an image to the carousel
        label = QLabel()
        label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setFixedSize(100, 100)
        self.inner_layout.addWidget(label)

    def clear_images(self):
        # Clear all images from the carousel
        while self.inner_layout.count():
            item = self.inner_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()