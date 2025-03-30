from PIL.ImageQt import QPixmap
from PySide6.QtGui import QImage, Qt
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

from component.FrameLabel import FrameLabel
from component.GlobalContext import gc
from component.ProcessFrameThread import ProcessFrameThread
from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.processFrameThread = ProcessFrameThread()

        gc.updateFrameSignal.connect(self.updateFrame)
        gc.startSignal.connect(self.startProcess)
        self.setup_ui()

    def setup_ui(self):
        self.main_layout = QVBoxLayout()
        self.origin_frame_label = QLabel("origin")
        self.origin_frame_label.setMinimumSize(640, 360)
        self.process_frame_label = FrameLabel("frame", self)
        self.process_frame_label.setMinimumSize(640, 360)
        self.main_layout.addWidget(self.process_frame_label)
        self.main_layout.addWidget(self.origin_frame_label)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setLayout(self.main_layout)

    def updateFrame(self, originFrame: QImage, processFrame: QImage):
        self.origin_frame_label.setPixmap(QPixmap().fromImage(originFrame))
        self.process_frame_label.setPixmap(QPixmap().fromImage(processFrame))

    def startProcess(self, flag: bool):
        if flag:
            self.processFrameThread.start()
        else:
            self.processFrameThread.stop()
