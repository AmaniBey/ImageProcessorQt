import sys

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QStackedWidget, QSplitter, QMenu, QListWidgetItem
)

from component.LeftSidebar import LeftSidebar
from component.MainWidget import MainWidget
from component.RightSidebar import RightSidebar
from component.ToolBar import ToolBar
from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class CustomMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 自定义布局 - QToolBar")
        self.setWindowFlags(Qt.WindowType.WindowMinimizeButtonHint |
                            Qt.WindowType.WindowMaximizeButtonHint |
                            Qt.WindowType.WindowCloseButtonHint)

        self.setWindowIcon(QIcon("https://i-blog.csdnimg.cn/blog_migrate/0b46d4fb50cd9a36b767c5782c718d97.png"))

        # 记录鼠标按下时的位置，用于拖动
        self.drag_position = QPoint()

        # ========== 1. 创建中央组件（包含主布局） ==========
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # ========== 2. 创建 QToolBar 作为自定义标题栏 ==========
        self.title_toolbar = ToolBar("自定义标题栏", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.title_toolbar)

        # ========== 3. 主内容区：QSplitter ==========
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # ========== 4. 左侧边栏（QDockWidget） ==========
        self.left_sidebar = LeftSidebar("Left Sidebar")
        self.left_sidebar.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_sidebar)

        # ========== 5. 右侧边栏（QDockWidget） ==========
        self.right_sidebar = RightSidebar("Right Sidebar")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_sidebar)

        # ========== 6. 主内容区（QStackedWidget） ==========
        self.main_content = QStackedWidget()

        self.page2 = MainWidget()
        self.main_content.addWidget(self.page2)
        self.splitter.addWidget(self.main_content)



    # 右键菜单

    # 切换页面
    def clicked_left_sidebar_item(self, item: QListWidgetItem):
        self.main_content.setCurrentIndex(self.left_sidebar_list.row(item))

    # ========== 8. 实现无边框下拖动窗口的功能 ==========
    def mousePressEvent(self, event):
        """
        鼠标按下时，如果在 title_toolbar 区域内，并且是左键，则记录此时鼠标与窗口左上角的相对位置。
        """
        if event.button() == Qt.MouseButton.LeftButton:
            # 使用 event.position() 代替 event.pos()
            widget_under_mouse = self.childAt(event.position().toPoint())
            if widget_under_mouse and (
                    widget_under_mouse is self.title_toolbar
                    or self.title_toolbar.isAncestorOf(widget_under_mouse)
            ):
                # 使用 event.globalPosition() 代替 event.globalPos()
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()

    def mouseMoveEvent(self, event):
        """
        鼠标移动时，若左键按下且已在 title_toolbar 区域开始拖动，则移动窗口。
        """
        if event.buttons() & Qt.MouseButton.LeftButton:
            if not self.drag_position.isNull():
                self.move(event.globalPosition().toPoint() - self.drag_position)
                event.accept()

    def mouseReleaseEvent(self, event):
        """
        鼠标释放后，重置拖动位置。
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = QPoint()
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setAttribute(Qt.ApplicationAttribute.AA_UseOpenGLES, True)  # Qt.AA_UseOpenGLES (在 PyQt6 中)

    window = CustomMainWindow()
    window.show()
    sys.exit(app.exec())
