import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog, QSlider, QCheckBox,
                               QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
                               QTabWidget, QGroupBox, QGridLayout)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal, Slot
from ultralytics import YOLO


class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.yolo_model = None

    def load_yolo_model(self, model_path='yolov8n.pt'):
        """加载YOLO模型"""
        try:
            self.yolo_model = YOLO(model_path)
            return True
        except Exception as e:
            print(f"加载YOLO模型错误: {e}")
            return False

    def load_image(self, image_path):
        """从给定路径加载图像"""
        try:
            self.original_image = cv2.imread(image_path)
            self.processed_image = self.original_image.copy()
            return True
        except Exception as e:
            print(f"加载图像错误: {e}")
            return False

    def reset_processing(self):
        """重置处理过的图像为原始图像"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            return True
        return False

    def resize_image(self, width, height):
        """将图像调整为给定尺寸"""
        if self.processed_image is None:
            return None
        try:
            self.processed_image = cv2.resize(self.processed_image, (width, height))
            return self.processed_image
        except Exception as e:
            print(f"调整图像尺寸错误: {e}")
            return None

    def convert_color(self, conversion_code):
        """将图像转换为不同的颜色空间"""
        if self.processed_image is None:
            return None
        try:
            self.processed_image = cv2.cvtColor(self.processed_image, conversion_code)
            return self.processed_image
        except Exception as e:
            print(f"转换颜色错误: {e}")
            return None

    def erode(self, kernel_size, iterations=3):
        """对图像应用腐蚀操作"""
        if self.processed_image is None:
            return None
        try:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.processed_image = cv2.erode(self.processed_image, kernel, iterations=iterations)
            return self.processed_image
        except Exception as e:
            print(f"应用腐蚀错误: {e}")
            return None

    def dilate(self, kernel_size, iterations=3):
        """对图像应用膨胀操作"""
        if self.processed_image is None:
            return None
        try:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.processed_image = cv2.dilate(self.processed_image, kernel, iterations=iterations)
            return self.processed_image
        except Exception as e:
            print(f"应用膨胀错误: {e}")
            return None

    def binary_threshold(self, threshold_value):
        """对图像应用二值化处理"""
        if self.processed_image is None:
            return None
        try:
            gray = self.processed_image
            if len(self.processed_image.shape) > 2:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            _, self.processed_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            return self.processed_image
        except Exception as e:
            print(f"应用二值化错误: {e}")
            return None

    def canny_edge_detection(self, threshold1, threshold2):
        """对图像应用Canny边缘检测"""
        if self.processed_image is None:
            return None
        try:
            gray = self.processed_image
            if len(self.processed_image.shape) > 2:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.Canny(gray, threshold1, threshold2)
            return self.processed_image
        except Exception as e:
            print(f"应用Canny边缘检测错误: {e}")
            return None

    def find_contours(self, min_area=0, max_area=float('inf')):
        """在图像中查找轮廓"""
        if self.processed_image is None:
            return None, None
        try:
            # 确保图像是二值的
            if len(self.processed_image.shape) > 2:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = self.processed_image

            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 过滤轮廓
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    filtered_contours.append(contour)

            # 绘制轮廓
            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if len(binary.shape) == 2 else self.processed_image.copy()
            cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)
            self.processed_image = result

            return filtered_contours, hierarchy
        except Exception as e:
            print(f"查找轮廓错误: {e}")
            return None, None

    def blur(self, kernel_size):
        """对图像应用均值模糊"""
        if self.processed_image is None:
            return None
        try:
            if kernel_size % 2 == 0:  # 确保内核大小是奇数
                kernel_size += 1
            self.processed_image = cv2.blur(self.processed_image, (kernel_size, kernel_size))
            return self.processed_image
        except Exception as e:
            print(f"应用均值模糊错误: {e}")
            return None

    def gaussian_blur(self, kernel_size):
        """对图像应用高斯模糊"""
        if self.processed_image is None:
            return None
        try:
            if kernel_size % 2 == 0:  # 确保内核大小是奇数
                kernel_size += 1
            self.processed_image = cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), 0)
            return self.processed_image
        except Exception as e:
            print(f"应用高斯模糊错误: {e}")
            return None

    def median_blur(self, kernel_size):
        """对图像应用中值模糊"""
        if self.processed_image is None:
            return None
        try:
            if kernel_size % 2 == 0:  # 确保内核大小是奇数
                kernel_size += 1
            self.processed_image = cv2.medianBlur(self.processed_image, kernel_size)
            return self.processed_image
        except Exception as e:
            print(f"应用中值模糊错误: {e}")
            return None

    def detect_objects_with_yolo(self):
        """使用YOLO模型检测对象"""
        if self.processed_image is None or self.yolo_model is None:
            return None

        try:
            # 使用YOLO进行检测
            results = self.yolo_model(self.processed_image)

            # 在图像上绘制检测结果
            result_image = results[0].plot()
            self.processed_image = result_image

            return results[0]
        except Exception as e:
            print(f"YOLO检测错误: {e}")
            return None


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.current_config_id = 1
        self.config_dir = Path("./conf")
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("图像处理工具")
        self.setMinimumSize(1200, 800)

        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建左侧面板 - 图像显示区
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 添加图像标签
        self.original_img_label = QLabel("原始图像")
        self.original_img_label.setAlignment(Qt.AlignCenter)
        self.original_img_label.setMinimumSize(500, 400)
        self.original_img_label.setStyleSheet("border: 1px solid #ccc;")

        self.processed_img_label = QLabel("处理后图像")
        self.processed_img_label.setAlignment(Qt.AlignCenter)
        self.processed_img_label.setMinimumSize(500, 400)
        self.processed_img_label.setStyleSheet("border: 1px solid #ccc;")

        # 创建按钮面板
        button_panel = QWidget()
        button_layout = QHBoxLayout(button_panel)

        self.load_image_btn = QPushButton("加载图像")
        self.load_image_btn.clicked.connect(self.load_image)

        self.save_image_btn = QPushButton("保存图像")
        self.save_image_btn.clicked.connect(self.save_image)

        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)

        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.clicked.connect(self.load_config)

        self.start_processing_btn = QPushButton("开始处理")
        self.start_processing_btn.clicked.connect(self.start_processing)

        self.reset_processing_btn = QPushButton("重置处理")
        self.reset_processing_btn.clicked.connect(self.reset_processing)

        button_layout.addWidget(self.load_image_btn)
        button_layout.addWidget(self.save_image_btn)
        button_layout.addWidget(self.save_config_btn)
        button_layout.addWidget(self.load_config_btn)
        button_layout.addWidget(self.start_processing_btn)
        button_layout.addWidget(self.reset_processing_btn)

        left_layout.addWidget(self.original_img_label)
        left_layout.addWidget(self.processed_img_label)
        left_layout.addWidget(button_panel)

        # 创建右侧面板 - 参数控制区
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        right_panel.setMinimumWidth(400)

        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)

        # 添加YOLO选择
        yolo_group = QGroupBox("YOLO检测设置")
        yolo_layout = QVBoxLayout()

        self.use_yolo_cb = QCheckBox("使用YOLO8")
        self.classes_file_path = QLineEdit()
        self.classes_file_btn = QPushButton("选择classes文件")
        self.classes_file_btn.clicked.connect(self.select_classes_file)

        yolo_layout.addWidget(self.use_yolo_cb)

        classes_layout = QHBoxLayout()
        classes_layout.addWidget(QLabel("Classes文件:"))
        classes_layout.addWidget(self.classes_file_path)
        classes_layout.addWidget(self.classes_file_btn)
        yolo_layout.addLayout(classes_layout)

        yolo_group.setLayout(yolo_layout)
        right_layout.addWidget(yolo_group)

        # 添加几何变换部分
        resize_group = QGroupBox("几何变换")
        resize_layout = QVBoxLayout()

        self.use_resize_cb = QCheckBox("应用缩放")

        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("宽度:"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 4000)
        self.width_spinbox.setValue(640)
        width_layout.addWidget(self.width_spinbox)

        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("高度:"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 4000)
        self.height_spinbox.setValue(480)
        height_layout.addWidget(self.height_spinbox)

        resize_layout.addWidget(self.use_resize_cb)
        resize_layout.addLayout(width_layout)
        resize_layout.addLayout(height_layout)
        resize_group.setLayout(resize_layout)
        right_layout.addWidget(resize_group)

        # 添加颜色空间转换部分
        color_group = QGroupBox("颜色空间转换")
        color_layout = QVBoxLayout()

        self.use_color_cb = QCheckBox("应用颜色转换")

        color_mode_layout = QHBoxLayout()
        color_mode_layout.addWidget(QLabel("转换模式:"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["BGR2RGB", "BGR2GRAY", "BGR2HSV", "RGB2BGR", "RGB2GRAY", "RGB2HSV"])
        color_mode_layout.addWidget(self.color_mode_combo)

        color_layout.addWidget(self.use_color_cb)
        color_layout.addLayout(color_mode_layout)
        color_group.setLayout(color_layout)
        right_layout.addWidget(color_group)

        # 添加形态学操作部分
        morph_group = QGroupBox("形态学操作")
        morph_layout = QVBoxLayout()

        # 腐蚀操作
        self.use_erode_cb = QCheckBox("应用腐蚀")

        erode_kernel_layout = QHBoxLayout()
        erode_kernel_layout.addWidget(QLabel("卷积核大小:"))
        self.erode_kernel_slider = QSlider(Qt.Horizontal)
        self.erode_kernel_slider.setRange(1, 20)
        self.erode_kernel_slider.setValue(5)
        erode_kernel_layout.addWidget(self.erode_kernel_slider)
        self.erode_kernel_label = QLabel("5")
        erode_kernel_layout.addWidget(self.erode_kernel_label)

        erode_iter_layout = QHBoxLayout()
        erode_iter_layout.addWidget(QLabel("迭代次数:"))
        self.erode_iter_spin = QSpinBox()
        self.erode_iter_spin.setRange(1, 10)
        self.erode_iter_spin.setValue(3)
        erode_iter_layout.addWidget(self.erode_iter_spin)

        # 膨胀操作
        self.use_dilate_cb = QCheckBox("应用膨胀")

        dilate_kernel_layout = QHBoxLayout()
        dilate_kernel_layout.addWidget(QLabel("卷积核大小:"))
        self.dilate_kernel_slider = QSlider(Qt.Horizontal)
        self.dilate_kernel_slider.setRange(1, 20)
        self.dilate_kernel_slider.setValue(5)
        dilate_kernel_layout.addWidget(self.dilate_kernel_slider)
        self.dilate_kernel_label = QLabel("5")
        dilate_kernel_layout.addWidget(self.dilate_kernel_label)

        dilate_iter_layout = QHBoxLayout()
        dilate_iter_layout.addWidget(QLabel("迭代次数:"))
        self.dilate_iter_spin = QSpinBox()
        self.dilate_iter_spin.setRange(1, 10)
        self.dilate_iter_spin.setValue(3)
        dilate_iter_layout.addWidget(self.dilate_iter_spin)

        # 二值化操作
        self.use_binary_cb = QCheckBox("应用二值化")

        binary_thresh_layout = QHBoxLayout()
        binary_thresh_layout.addWidget(QLabel("阈值:"))
        self.binary_thresh_slider = QSlider(Qt.Horizontal)
        self.binary_thresh_slider.setRange(0, 255)
        self.binary_thresh_slider.setValue(127)
        binary_thresh_layout.addWidget(self.binary_thresh_slider)
        self.binary_thresh_label = QLabel("127")
        binary_thresh_layout.addWidget(self.binary_thresh_label)

        # Canny边缘检测
        self.use_canny_cb = QCheckBox("应用Canny边缘检测")

        canny_thresh1_layout = QHBoxLayout()
        canny_thresh1_layout.addWidget(QLabel("阈值1:"))
        self.canny_thresh1_slider = QSlider(Qt.Horizontal)
        self.canny_thresh1_slider.setRange(0, 255)
        self.canny_thresh1_slider.setValue(100)
        canny_thresh1_layout.addWidget(self.canny_thresh1_slider)
        self.canny_thresh1_label = QLabel("100")
        canny_thresh1_layout.addWidget(self.canny_thresh1_label)

        canny_thresh2_layout = QHBoxLayout()
        canny_thresh2_layout.addWidget(QLabel("阈值2:"))
        self.canny_thresh2_slider = QSlider(Qt.Horizontal)
        self.canny_thresh2_slider.setRange(0, 255)
        self.canny_thresh2_slider.setValue(200)
        canny_thresh2_layout.addWidget(self.canny_thresh2_slider)
        self.canny_thresh2_label = QLabel("200")
        canny_thresh2_layout.addWidget(self.canny_thresh2_label)

        morph_layout.addWidget(self.use_erode_cb)
        morph_layout.addLayout(erode_kernel_layout)
        morph_layout.addLayout(erode_iter_layout)
        morph_layout.addWidget(self.use_dilate_cb)
        morph_layout.addLayout(dilate_kernel_layout)
        morph_layout.addLayout(dilate_iter_layout)
        morph_layout.addWidget(self.use_binary_cb)
        morph_layout.addLayout(binary_thresh_layout)
        morph_layout.addWidget(self.use_canny_cb)
        morph_layout.addLayout(canny_thresh1_layout)
        morph_layout.addLayout(canny_thresh2_layout)
        morph_group.setLayout(morph_layout)
        right_layout.addWidget(morph_group)

        # 添加图像滤波部分
        filter_group = QGroupBox("图像滤波")
        filter_layout = QVBoxLayout()

        # 均值模糊
        self.use_blur_cb = QCheckBox("应用均值模糊")

        blur_kernel_layout = QHBoxLayout()
        blur_kernel_layout.addWidget(QLabel("核大小:"))
        self.blur_kernel_slider = QSlider(Qt.Horizontal)
        self.blur_kernel_slider.setRange(1, 30)
        self.blur_kernel_slider.setValue(5)
        blur_kernel_layout.addWidget(self.blur_kernel_slider)
        self.blur_kernel_label = QLabel("5")
        blur_kernel_layout.addWidget(self.blur_kernel_label)

        # 高斯模糊
        self.use_gaussian_cb = QCheckBox("应用高斯模糊")

        gaussian_kernel_layout = QHBoxLayout()
        gaussian_kernel_layout.addWidget(QLabel("核大小:"))
        self.gaussian_kernel_slider = QSlider(Qt.Horizontal)
        self.gaussian_kernel_slider.setRange(1, 30)
        self.gaussian_kernel_slider.setValue(5)
        gaussian_kernel_layout.addWidget(self.gaussian_kernel_slider)
        self.gaussian_kernel_label = QLabel("5")
        gaussian_kernel_layout.addWidget(self.gaussian_kernel_label)

        # 中值滤波
        self.use_median_cb = QCheckBox("应用中值滤波")

        median_kernel_layout = QHBoxLayout()
        median_kernel_layout.addWidget(QLabel("核大小:"))
        self.median_kernel_slider = QSlider(Qt.Horizontal)
        self.median_kernel_slider.setRange(1, 30)
        self.median_kernel_slider.setValue(5)
        median_kernel_layout.addWidget(self.median_kernel_slider)
        self.median_kernel_label = QLabel("5")
        median_kernel_layout.addWidget(self.median_kernel_label)

        filter_layout.addWidget(self.use_blur_cb)
        filter_layout.addLayout(blur_kernel_layout)
        filter_layout.addWidget(self.use_gaussian_cb)
        filter_layout.addLayout(gaussian_kernel_layout)
        filter_layout.addWidget(self.use_median_cb)
        filter_layout.addLayout(median_kernel_layout)
        filter_group.setLayout(filter_layout)
        right_layout.addWidget(filter_group)

        # 连接滑块值变化信号
        self.erode_kernel_slider.valueChanged.connect(
            lambda v: self.erode_kernel_label.setText(str(v)))
        self.dilate_kernel_slider.valueChanged.connect(
            lambda v: self.dilate_kernel_label.setText(str(v)))
        self.binary_thresh_slider.valueChanged.connect(
            lambda v: self.binary_thresh_label.setText(str(v)))
        self.canny_thresh1_slider.valueChanged.connect(
            lambda v: self.canny_thresh1_label.setText(str(v)))
        self.canny_thresh2_slider.valueChanged.connect(
            lambda v: self.canny_thresh2_label.setText(str(v)))
        self.blur_kernel_slider.valueChanged.connect(
            lambda v: self.blur_kernel_label.setText(str(v)))
        self.gaussian_kernel_slider.valueChanged.connect(
            lambda v: self.gaussian_kernel_label.setText(str(v)))
        self.median_kernel_slider.valueChanged.connect(
            lambda v: self.median_kernel_label.setText(str(v)))

        right_panel.setWidget(right_content)

        # 添加面板到主布局
        main_layout.addWidget(left_panel, 3)  # 左侧占比较大
        main_layout.addWidget(right_panel, 2)  # 右侧占比较小

        # 创建配置目录
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)

    def load_image(self):
        """打开文件对话框加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            if self.processor.load_image(file_path):
                self.display_images()
                self.start_processing_btn.setEnabled(True)
                self.reset_processing_btn.setEnabled(True)
            else:
                # 显示错误消息
                pass

    def save_image(self):
        """保存处理后的图像"""
        if self.processor.processed_image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", "PNG文件 (*.png);;JPG文件 (*.jpg);;所有文件 (*)"
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.processor.processed_image)
            except Exception as e:
                print(f"保存图像错误: {e}")

    def save_config(self):
        """保存当前配置到JSON文件"""
        config = {
            "width": self.width_spinbox.value(),
            "height": self.height_spinbox.value(),
            "use_resize": self.use_resize_cb.isChecked(),
            "use_color": self.use_color_cb.isChecked(),
            "color_mode": self.color_mode_combo.currentText(),
            "use_erode": self.use_erode_cb.isChecked(),
            "erode_kernel": self.erode_kernel_slider.value(),
            "erode_iter": self.erode_iter_spin.value(),
            "use_dilate": self.use_dilate_cb.isChecked(),
            "dilate_kernel": self.dilate_kernel_slider.value(),
            "dilate_iter": self.dilate_iter_spin.value(),
            "use_binary": self.use_binary_cb.isChecked(),
            "binary_thresh": self.binary_thresh_slider.value(),
            "use_canny": self.use_canny_cb.isChecked(),
            "canny_thresh1": self.canny_thresh1_slider.value(),
            "canny_thresh2": self.canny_thresh2_slider.value(),
            "use_blur": self.use_blur_cb.isChecked(),
            "blur_kernel": self.blur_kernel_slider.value(),
            "use_gaussian": self.use_gaussian_cb.isChecked(),
            "gaussian_kernel": self.gaussian_kernel_slider.value(),
            "use_median": self.use_median_cb.isChecked(),
            "median_kernel": self.median_kernel_slider.value(),
            "use_yolo": self.use_yolo_cb.isChecked(),
            "classes_file": self.classes_file_path.text()
        }

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存配置", str(self.config_dir), "JSON文件 (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)
            except Exception as e:
                print(f"保存配置错误: {e}")

    def load_config(self):
        """从JSON文件加载配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载配置", str(self.config_dir), "JSON文件 (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)

                # 更新UI控件值
                self.width_spinbox.setValue(config.get("width", 640))
                self.height_spinbox.setValue(config.get("height", 480))
                self.use_resize_cb.setChecked(config.get("use_resize", False))
                self.use_color_cb.setChecked(config.get("use_color", False))

                color_mode = config.get("color_mode", "BGR2RGB")
                index = self.color_mode_combo.findText(color_mode)
                if index >= 0:
                    self.color_mode_combo.setCurrentIndex(index)

                self.use_erode_cb.setChecked(config.get("use_erode", False))
                self.erode_kernel_slider.setValue(config.get("erode_kernel", 5))
                self.erode_iter_spin.setValue(config.get("erode_iter", 3))

                self.use_dilate_cb.setChecked(config.get("use_dilate", False))
                self.dilate_kernel_slider.setValue(config.get("dilate_kernel", 5))
                self.dilate_iter_spin.setValue(config.get("dilate_iter", 3))

                self.use_binary_cb.setChecked(config.get("use_binary", False))
                self.binary_thresh_slider.setValue(config.get("binary_thresh", 127))

                self.use_canny_cb.setChecked(config.get("use_canny", False))
                self.canny_thresh1_slider.setValue(config.get("canny_thresh1", 100))
                self.canny_thresh2_slider.setValue(config.get("canny_thresh2", 200))

                self.use_blur_cb.setChecked(config.get("use_blur", False))
                self.blur_kernel_slider.setValue(config.get("blur_kernel", 5))

                self.use_gaussian_cb.setChecked(config.get("use_gaussian", False))
                self.gaussian_kernel_slider.setValue(config.get("gaussian_kernel", 5))

                self.use_median_cb.setChecked(config.get("use_median", False))
                self.median_kernel_slider.setValue(config.get("median_kernel", 5))

                self.use_yolo_cb.setChecked(config.get("use_yolo", False))
                self.classes_file_path.setText(config.get("classes_file", ""))

                # 如果已经加载图像，则重新应用处理
                if self.processor.original_image is not None:
                    self.start_processing()

            except Exception as e:
                print(f"加载配置错误: {e}")

    def select_classes_file(self):
        """选择classes文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Classes文件", "", "文本文件 (*.txt)"
        )

        if file_path:
            self.classes_file_path.setText(file_path)

    def start_processing(self):
        """应用所有启用的处理操作到图像"""
        if self.processor.original_image is None:
            return

        # 重置处理状态
        self.processor.reset_processing()

        # 应用所有启用的处理
        if self.use_resize_cb.isChecked():
            width = self.width_spinbox.value()
            height = self.height_spinbox.value()
            self.processor.resize_image(width, height)

        if self.use_color_cb.isChecked():
            color_mode = self.color_mode_combo.currentText()
            conversion_code = getattr(cv2, "COLOR_" + color_mode)
            self.processor.convert_color(conversion_code)

        if self.use_erode_cb.isChecked():
            kernel_size = self.erode_kernel_slider.value()
            iterations = self.erode_iter_spin.value()
            self.processor.erode(kernel_size, iterations)

        if self.use_dilate_cb.isChecked():
            kernel_size = self.dilate_kernel_slider.value()
            iterations = self.dilate_iter_spin.value()
            self.processor.dilate(kernel_size, iterations)

        if self.use_binary_cb.isChecked():
            threshold = self.binary_thresh_slider.value()
            self.processor.binary_threshold(threshold)

        if self.use_canny_cb.isChecked():
            threshold1 = self.canny_thresh1_slider.value()
            threshold2 = self.canny_thresh2_slider.value()
            self.processor.canny_edge_detection(threshold1, threshold2)

        if self.use_blur_cb.isChecked():
            kernel_size = self.blur_kernel_slider.value()
            self.processor.blur(kernel_size)

        if self.use_gaussian_cb.isChecked():
            kernel_size = self.gaussian_kernel_slider.value()
            self.processor.gaussian_blur(kernel_size)

        if self.use_median_cb.isChecked():
            kernel_size = self.median_kernel_slider.value()
            self.processor.median_blur(kernel_size)

        if self.use_yolo_cb.isChecked():
            if self.processor.yolo_model is None:
                self.processor.load_yolo_model()
            self.processor.detect_objects_with_yolo()

        # 更新显示
        self.display_images()

    def reset_processing(self):
        """重置图像处理"""
        if self.processor.reset_processing():
            self.display_images()

    def display_images(self):
        """在UI中显示原始图像和处理后的图像"""
        # 显示原始图像
        if self.processor.original_image is not None:
            h, w, c = self.processor.original_image.shape if len(self.processor.original_image.shape) == 3 else (
            *self.processor.original_image.shape, 1)

            if c == 1:  # 灰度图像
                q_img = QImage(self.processor.original_image.data, w, h, w, QImage.Format_Grayscale8)
            else:  # BGR彩色图像
                rgb_image = cv2.cvtColor(self.processor.original_image, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_image.data, w, h, w * c, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.original_img_label.width(), self.original_img_label.height(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_img_label.setPixmap(pixmap)

        # 显示处理后的图像
        if self.processor.processed_image is not None:
            h, w = self.processor.processed_image.shape[:2]
            c = 1 if len(self.processor.processed_image.shape) == 2 else self.processor.processed_image.shape[2]

            if c == 1:  # 灰度图像
                q_img = QImage(self.processor.processed_image.data, w, h, w, QImage.Format_Grayscale8)
            else:  # BGR彩色图像
                rgb_image = cv2.cvtColor(self.processor.processed_image, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_image.data, w, h, w * c, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.processed_img_label.width(), self.processed_img_label.height(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.processed_img_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec())
