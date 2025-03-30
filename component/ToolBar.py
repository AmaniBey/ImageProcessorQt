from PySide6.QtWidgets import QToolBar, QLabel, QComboBox, QPushButton, QFileDialog

from component.ConfigDialog import ConfigDialog
from component.GlobalContext import gc
from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class ToolBar(QToolBar):
    def __init__(self, parent=..., f=...):
        super().__init__(parent, f)
        self.parent = parent
        self.setMovable(False)
        self.set_ui()

    def set_ui(self):

        # 添加输入源选择控件
        camera_label = QLabel("输入源:")
        self.addWidget(camera_label)

        camera_combo = QComboBox()
        camera_combo.addItems(["摄像头", "视频文件", "图片"])
        camera_combo.currentIndexChanged.connect(self.on_source_changed)
        self.addWidget(camera_combo)

        # 摄像头索引选择
        camera_index_combo = QComboBox()
        camera_index_combo.addItems([f"{i}" for i in range(10)])

        camera_index_combo.currentIndexChanged.connect(lambda index: gc.setCapIndex(index))
        self.addWidget(camera_index_combo)

        select_video_button = QPushButton("选择视频")
        select_video_button.setObjectName("select_video_button")
        select_video_button.clicked.connect(self.choose_file)

        select_image_button = QPushButton("选择image")
        select_image_button.setObjectName("select_image_button")
        select_image_button.clicked.connect(self.choose_file)

        start_button = QPushButton("开始")
        start_button.clicked.connect(lambda: gc.start(True))
        save_button = QPushButton("保存配置")
        save_button.clicked.connect(self.save_config)

        self.addWidget(select_video_button)
        self.addWidget(select_image_button)
        self.addWidget(start_button)
        self.addWidget(save_button)

    def save_config(self):
        # 弹出表单
        dialog = ConfigDialog(execute=gc.cf.save_conf_by_path)
        if dialog.exec():
            logger.info("保存配置成功")

    def on_source_changed(self, index):
        """处理输入源选择变化"""
        gc.openmodel = index
        logger.info(f"输入源选择变化: {index}")

    def choose_file(self):
        sender = self.sender()
        """选择文件对话框"""
        if sender.objectName() == "select_video_button":
            file_filter = "视频文件 (*.mp4 *.avi);;"
            file_path, _ = QFileDialog.getOpenFileName(self, f"选择文件", "", file_filter)
            if file_path:
                gc.current_video_path = file_path
            logger.info(f"选择文件: {file_path}")
        else:
            file_filter = "图片文件 (*.jpg *.png *.jpeg);;"
            file_path, _ = QFileDialog.getOpenFileName(self, f"选择文件", "", file_filter)
            if file_path:
                gc.current_image_path = file_path
            logger.info(f"选择文件: {file_path}")
