import csv
import pathlib
from enum import Enum

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QAction, QPainter
from PySide6.QtWidgets import QLabel, QMenu

from component.ConfigDialog import ConfigDialog
from component.GlobalContext import gc


class DRAW_MODEL(Enum):
    DO_NOTHING = 0
    EXCLUSION = 1
    CANCEL_EXCLUSION = 2
    LOAD_EXCLUSION = 3


class FrameLabel(QLabel):
    points_changed = Signal(list)  # 信号：发送最终点集

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.parent = parent
        self.draw_model = DRAW_MODEL.DO_NOTHING
        self.points = []  # 存储所有点
        self.is_drawing = False  # 是否正在绘制
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setMouseTracking(True)  # 启用鼠标追踪

    def show_context_menu(self, pos):
        menu = QMenu(self)
        action1 = QAction("绘制排除区域")
        action1.triggered.connect(lambda: self.set_draw_model(DRAW_MODEL.EXCLUSION))
        action2 = QAction("绘制取消排除区域")
        action2.triggered.connect(lambda: self.set_draw_model(DRAW_MODEL.CANCEL_EXCLUSION))
        menu.addActions([action1, action2])

        subMenu = QMenu("加载排除区域")
        for item in gc.get_exclusion_file():
            action3 = QAction(item["name"])
            action3.triggered.connect(lambda: self.load_points_from_csv(item["path"]))
            subMenu.addAction(action3)

        subMenu.addMenu(subMenu)
        menu.addMenu(subMenu)
        menu.exec(self.mapToGlobal(pos))

    def set_draw_model(self, mode):
        self.draw_model = mode
        print("按住鼠标左键拖动绘制，松开结束")

    def mousePressEvent(self, event):
        if self.draw_model == DRAW_MODEL.EXCLUSION and event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.points.append(event.pos())  # 记录起点
            self.update()  # 触发重绘
        if self.draw_model == DRAW_MODEL.CANCEL_EXCLUSION and event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.points.append(event.pos())  # 记录起点
            self.update()  # 触发重绘

    def mouseMoveEvent(self, event):
        if self.draw_model == DRAW_MODEL.EXCLUSION and self.is_drawing:
            self.points.append(event.pos())  # 持续记录移动轨迹
            self.update()  # 实时更新绘制
            gc.drawMaskSignale.emit(self.points)

        elif self.draw_model == DRAW_MODEL.CANCEL_EXCLUSION and self.is_drawing:
            cursor_pos = event.pos()
            cancel_radius = 20
            points_to_remove = []
            for point in self.points:
                distance = (cursor_pos.x() - point.x()) ** 2 + (cursor_pos.y() - point.y()) ** 2
                if distance <= cancel_radius ** 2:
                    points_to_remove.append(point)

            for point in points_to_remove:
                if point in self.points:
                    self.points.remove(point)
            if points_to_remove:
                self.update()
                gc.drawMaskSignale.emit(self.points)  # 发送更新后的点集

    def mouseReleaseEvent(self, event):
        if self.draw_model == DRAW_MODEL.EXCLUSION and event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = False
            if len(self.points) >= 3:  # 至少3个点才能构成区域
                self.points_changed.emit(self.points)  # 发送点集
            self.draw_model = DRAW_MODEL.DO_NOTHING  # 结束绘制

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.draw_model == DRAW_MODEL.EXCLUSION and self.points:
            painter = QPainter(self)

            # 绘制所有点
            for point in self.points:
                # 实心的圆
                painter.drawPoint(point)

    def save_points_to_csv(self):
        """将 self.points 保存为 CSV 文件"""
        dialog = ConfigDialog(self, self.save_points_to_csv)
        pass
        # with open(filename, 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["x", "y"])  # 表头
        #     for point in self.points:
        #         writer.writerow([point.x(), point.y()])

    def load_points_from_csv(self, filename):
        """从 CSV 文件加载点集"""
        self.points = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                x, y = map(int, row)
                self.points.append(QPoint(x, y))
        self.update()
