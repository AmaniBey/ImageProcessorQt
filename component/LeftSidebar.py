from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QVBoxLayout, QLabel, QComboBox, QWidget

from component.GlobalContext import gc
from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class LeftSidebar(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)

        # 存储当前选择的配置
        self.current_basic_config = "基础配置1"
        self.current_color_config = "颜色配置1"

        self.setup_ui()

    def setup_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(self)
        # 标题
        title_label = QLabel("配置信息")
        layout.addWidget(title_label)

        # 基础配置
        layout.addSpacing(10)
        self.basic_config_combo = QComboBox()
        configfiles = gc.get_configfile()
        for configfile in configfiles:
            self.basic_config_combo.addItem(configfile["name"], configfile["path"])
        self.basic_config_combo.currentTextChanged.connect(self.on_basic_config_changed)
        layout.addWidget(self.basic_config_combo)

        # 颜色配置
        layout.addSpacing(10)
        color_label = QLabel("颜色配置")
        layout.addWidget(color_label)
        self.color_config_combo = QComboBox()
        self.color_config_combo.addItems(["颜色配置1", "颜色配置2", "颜色配置3"])
        self.color_config_combo.currentTextChanged.connect(self.on_color_config_changed)
        layout.addWidget(self.color_config_combo)

        # 添加弹性空间
        layout.addStretch()
        widget.setLayout(layout)
        self.setWidget(widget)

    def on_basic_config_changed(self, text):
        """处理基础配置选择变化"""
        self.current_basic_config = text
        # 这里可以添加配置变更后的处理逻辑
        filepath = self.basic_config_combo.currentData()
        gc.cf.config_file = filepath
        gc.cf.load_conf()
        gc.updateConfigSignale.emit(filepath)


    def on_color_config_changed(self, text):
        """处理颜色配置选择变化"""
        self.current_color_config = text
        # 这里可以添加配置变更后的处理逻辑

    def get_current_config(self):
        """获取当前配置信息"""
        return {
            'basic_config': self.current_basic_config,
            'color_config': self.current_color_config
        }
