from PySide6.QtWidgets import QVBoxLayout, QWidget, QScrollArea, \
    QPushButton

from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class ColorTab(QWidget):

    def __init__(self):
        super().__init__()
        # 滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.widget_area = QWidget()
        self.scroll_area.setWidget(self.widget_area)

        self.main_layout = QVBoxLayout()
        self.widget_area.setLayout(self.main_layout)

        vbox = QVBoxLayout()
        vbox.addWidget(self.scroll_area)
        self.setLayout(vbox)
        self.setup_ui()

    def setup_ui(self):
        for i in range(100):
            self.main_layout.addWidget(QPushButton("test"))
