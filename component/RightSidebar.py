from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QTabWidget

from component.ColorTab import ColorTab
from component.ConfigTab import ConfigTab
from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class RightSidebar(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.setup_ui()

    def setup_ui(self):
        self.tab_widget = QTabWidget()
        # 创建选项卡页面
        self.configTab = ConfigTab()
        self.coloTab = ColorTab()
        # 添加选项卡
        self.tab_widget.addTab(self.configTab, "配置信息")
        self.tab_widget.addTab(self.coloTab, "颜色配置")
        self.setWidget(self.tab_widget)
