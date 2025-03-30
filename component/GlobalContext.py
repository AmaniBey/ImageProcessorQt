from pathlib import Path
from typing import List

import cv2
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from utils.ColoredFormatter import GlobalLogger
from utils.Configuration import Configuration


class GlobalContext(QObject):
    startSignal = Signal(bool)
    updateConfigSignale = Signal(object)
    drawMaskSignale = Signal(object)
    updateFrameSignal = Signal(QImage, QImage)


    def __init__(self):
        super().__init__()

        self.cap = None
        self.cap_index = 1
        self.openmodel = 0  # 0 摄像头, 1 视频, 2 图片
        self.current_video_path = ""
        self.current_image_path = ""
        self.target_width = 640
        self.target_height = 360

        self.configlist = [{}] # 配置文件列表

        self.knn = cv2.createBackgroundSubtractorKNN()
        self.mog2 = cv2.createBackgroundSubtractorMOG2()

        self.cf: Configuration = Configuration()

    def start(self, flag: bool):
        if flag:
            if self.openmodel == 0:
                self.cap = cv2.VideoCapture(self.cap_index)
            elif self.openmodel == 1:
                self.cap = cv2.VideoCapture(self.current_video_path)
                # 无限循环

            elif self.openmodel == 2:
                self.cap = cv2.VideoCapture(self.current_image_path)

        self.startSignal.emit(flag)

        print(f"start {flag} open model {self.openmodel}")

    def setCapIndex(self, index: int):
        self.cap_index = index
        print(f"select camera {index}")

    def setOpenModel(self, index: int):
        self.openmodel = index
        print(f"select open model {index}")

    def get_configfile(self)->List[dict]:
        # 得到当前项目目录下的./conf 所有*.json文件
        conf_dir = Path.cwd() / 'conf'
        self.configlist = []
        for file in conf_dir.glob('*.json'):
            if file.is_file():
                # 获取文件名称,取消后缀
                item = {
                    "path": str(file),
                    "name": file.stem
                }
                self.configlist.append(item)
        return self.configlist
    def add_configfile(self,name):
        self.configlist.append(name)


    def get_exclusion_file(self):
        conf_dir = Path.cwd() / 'conf'
        config_list = []
        for file in conf_dir.glob('*.csv'):
            if file.is_file():
                item = {
                    "path": str(file),
                    "name": file.stem
                }
                config_list.append(item)
        return config_list

gc = GlobalContext()
