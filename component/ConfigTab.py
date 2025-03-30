from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QLabel, QComboBox, QWidget, QScrollArea, \
    QGroupBox, QCheckBox, QHBoxLayout, QSpinBox, QSlider, QFormLayout

from component.GlobalContext import gc
from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class ConfigTab(QWidget):

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

        self.set_config()
        self._connect_signals()

        gc.updateConfigSignale.connect(self.set_config)

    def _connect_signals(self):
        # 获取所有需要监听的控件类型
        widget_signal_map = {
            QSlider: 'valueChanged',
            QSpinBox: 'valueChanged',
            QCheckBox: 'stateChanged',
            QComboBox: 'currentIndexChanged'
        }

        # 遍历子组件并连接信号
        for child in self.findChildren(QWidget):
            for widget_type, signal in widget_signal_map.items():
                if isinstance(child, widget_type):
                    getattr(child, signal).connect(self.get_config)

    def setup_ui(self):
        self.background_group()

        # 几何变换部分
        self.create_geometry_transform_group()

        # 颜色空间转换部分
        self.create_color_space_group()

        # 形态学操作部分
        self.create_morphology_group()

        # 图像滤波部分
        self.create_filter_group()

        # 添加一个弹性空间，确保控件靠上显示
        self.main_layout.addStretch()

    def create_geometry_transform_group(self):
        group_box = QGroupBox("几何变换")
        layout = QVBoxLayout()

        # 启用缩放复选框
        self.resize_checkbox = QCheckBox("启用缩放")
        layout.addWidget(self.resize_checkbox)

        # 宽度控制
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("宽度："))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 4000)
        self.width_spinbox.setValue(640)
        width_layout.addWidget(self.width_spinbox)
        layout.addLayout(width_layout)

        # 高度控制
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("高度："))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 4000)
        self.height_spinbox.setValue(480)
        height_layout.addWidget(self.height_spinbox)
        layout.addLayout(height_layout)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def create_color_space_group(self):
        group_box = QGroupBox("颜色空间转换")
        layout = QVBoxLayout()

        # 启用颜色转换复选框
        self.color_convert_checkbox = QCheckBox("启用颜色转换")
        layout.addWidget(self.color_convert_checkbox)

        # 转换模式下拉框
        layout.addWidget(QLabel("转换模式："))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["BGR2GRAY", "BGR2RGB", "BGR2HSV", "RGB2BGR"])
        layout.addWidget(self.color_mode_combo)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def create_morphology_group(self):
        group_box = QGroupBox("形态学操作")
        layout = QVBoxLayout()

        # 开运算
        self.open_checkbox = QCheckBox("启用开运算")
        layout.addWidget(self.open_checkbox)

        open_kernel_layout = QVBoxLayout()
        open_kernel_layout.addWidget(QLabel("卷积核大小："))
        self.open_kernel_slider = QSlider(Qt.Horizontal)
        self.open_kernel_slider.setRange(1, 20)
        self.open_kernel_slider.setValue(5)
        open_kernel_layout.addWidget(self.open_kernel_slider)
        layout.addLayout(open_kernel_layout)

        self.open_iter_spinbox = QSpinBox()
        self.open_iter_spinbox.setRange(1, 10)
        self.open_iter_spinbox.setValue(3)
        open_iter_layout = QHBoxLayout()
        open_iter_layout.addWidget(QLabel("迭代次数："))
        open_iter_layout.addWidget(self.open_iter_spinbox)
        layout.addLayout(open_iter_layout)

        # 闭运算
        self.close_checkbox = QCheckBox("启用闭运算")
        layout.addWidget(self.close_checkbox)
        close_kernel_layout = QVBoxLayout()
        close_kernel_layout.addWidget(QLabel("卷积核大小："))
        self.close_kernel_slider = QSlider(Qt.Horizontal)
        self.close_kernel_slider.setRange(1, 20)
        self.close_kernel_slider.setValue(5)
        close_kernel_layout.addWidget(self.close_kernel_slider)
        layout.addLayout(close_kernel_layout)
        self.close_iter_spinbox = QSpinBox()
        self.close_iter_spinbox.setRange(1, 10)
        self.close_iter_spinbox.setValue(3)
        close_iter_layout = QHBoxLayout()
        close_iter_layout.addWidget(QLabel("迭代次数："))
        close_iter_layout.addWidget(self.close_iter_spinbox)
        layout.addLayout(close_iter_layout)

        # 腐蚀操作
        self.erode_checkbox = QCheckBox("启用腐蚀")
        layout.addWidget(self.erode_checkbox)

        erode_kernel_layout = QVBoxLayout()
        erode_kernel_layout.addWidget(QLabel("卷积核大小："))
        self.erode_kernel_slider = QSlider(Qt.Horizontal)
        self.erode_kernel_slider.setRange(1, 20)
        self.erode_kernel_slider.setValue(5)
        erode_kernel_layout.addWidget(self.erode_kernel_slider)
        layout.addLayout(erode_kernel_layout)

        erode_iter_layout = QHBoxLayout()
        erode_iter_layout.addWidget(QLabel("迭代次数："))
        self.erode_iter_spinbox = QSpinBox()
        self.erode_iter_spinbox.setRange(1, 10)
        self.erode_iter_spinbox.setValue(3)
        erode_iter_layout.addWidget(self.erode_iter_spinbox)
        layout.addLayout(erode_iter_layout)

        # 膨胀操作
        self.dilate_checkbox = QCheckBox("启用膨胀")
        layout.addWidget(self.dilate_checkbox)

        dilate_kernel_layout = QVBoxLayout()
        dilate_kernel_layout.addWidget(QLabel("卷积核大小："))
        self.dilate_kernel_slider = QSlider(Qt.Horizontal)
        self.dilate_kernel_slider.setRange(1, 20)
        self.dilate_kernel_slider.setValue(5)
        dilate_kernel_layout.addWidget(self.dilate_kernel_slider)
        layout.addLayout(dilate_kernel_layout)

        dilate_iter_layout = QHBoxLayout()
        dilate_iter_layout.addWidget(QLabel("迭代次数："))
        self.dilate_iter_spinbox = QSpinBox()
        self.dilate_iter_spinbox.setRange(1, 10)
        self.dilate_iter_spinbox.setValue(3)
        dilate_iter_layout.addWidget(self.dilate_iter_spinbox)
        layout.addLayout(dilate_iter_layout)

        # 二值化操作
        self.binary_checkbox = QCheckBox("启用二值化")
        layout.addWidget(self.binary_checkbox)

        binary_threshold_layout = QVBoxLayout()
        binary_threshold_layout.addWidget(QLabel("阈值："))
        self.binary_threshold_slider = QSlider(Qt.Horizontal)
        self.binary_threshold_slider.setRange(0, 255)
        self.binary_threshold_slider.setValue(127)
        binary_threshold_layout.addWidget(self.binary_threshold_slider)
        layout.addLayout(binary_threshold_layout)

        # Canny边缘检测
        self.canny_checkbox = QCheckBox("启用Canny边缘检测")
        layout.addWidget(self.canny_checkbox)

        canny_threshold1_layout = QVBoxLayout()
        canny_threshold1_layout.addWidget(QLabel("阈值1："))
        self.canny_threshold1_slider = QSlider(Qt.Horizontal)
        self.canny_threshold1_slider.setRange(0, 255)
        self.canny_threshold1_slider.setValue(100)
        canny_threshold1_layout.addWidget(self.canny_threshold1_slider)
        layout.addLayout(canny_threshold1_layout)

        canny_threshold2_layout = QVBoxLayout()
        canny_threshold2_layout.addWidget(QLabel("阈值2："))
        self.canny_threshold2_slider = QSlider(Qt.Horizontal)
        self.canny_threshold2_slider.setRange(0, 255)
        self.canny_threshold2_slider.setValue(200)
        canny_threshold2_layout.addWidget(self.canny_threshold2_slider)
        layout.addLayout(canny_threshold2_layout)

        # 轮廓检测 - 完善版本
        self.contour_checkbox = QCheckBox("启用轮廓检测")
        self.contour_checkbox.setChecked(False)  # 默认不启用
        layout.addWidget(self.contour_checkbox)

        min_area_layout = QHBoxLayout()
        min_area_layout.addWidget(QLabel("最小面积："))
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(1, 10000)
        self.min_area_spinbox.setValue(103)
        min_area_layout.addWidget(self.min_area_spinbox)
        layout.addLayout(min_area_layout)

        max_area_layout = QHBoxLayout()
        max_area_layout.addWidget(QLabel("最大面积："))
        self.max_area_spinbox = QSpinBox()
        self.max_area_spinbox.setRange(1, 100000)
        self.max_area_spinbox.setValue(10000)
        max_area_layout.addWidget(self.max_area_spinbox)
        layout.addLayout(max_area_layout)

        # 添加轮廓拟合选项
        contour_approx_layout = QHBoxLayout()
        contour_approx_layout.addWidget(QLabel("轮廓拟合精度："))
        self.contour_approx_slider = QSlider(Qt.Horizontal)
        self.contour_approx_slider.setRange(1, 100)
        self.contour_approx_slider.setValue(10)  # 默认值，对应0.1的epsilon
        contour_approx_layout.addWidget(self.contour_approx_slider)
        layout.addLayout(contour_approx_layout)

        # 轮廓检测模式
        contour_mode_layout = QHBoxLayout()
        contour_mode_layout.addWidget(QLabel("轮廓检测模式："))
        self.contour_mode_combo = QComboBox()
        self.contour_mode_combo.addItems(["RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE"])
        contour_mode_layout.addWidget(self.contour_mode_combo)
        layout.addLayout(contour_mode_layout)

        # 轮廓近似方法
        contour_method_layout = QHBoxLayout()
        contour_method_layout.addWidget(QLabel("轮廓近似方法："))
        self.contour_method_combo = QComboBox()
        self.contour_method_combo.addItems(["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE"])
        contour_method_layout.addWidget(self.contour_method_combo)
        layout.addLayout(contour_method_layout)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def create_filter_group(self):
        group_box = QGroupBox("图像滤波")
        layout = QVBoxLayout()

        # 均值滤波
        self.blur_checkbox = QCheckBox("启用均值滤波")
        layout.addWidget(self.blur_checkbox)

        blur_kernel_layout = QVBoxLayout()
        blur_kernel_layout.addWidget(QLabel("卷积核大小："))
        self.blur_kernel_slider = QSlider(Qt.Horizontal)
        self.blur_kernel_slider.setRange(1, 30)
        self.blur_kernel_slider.setValue(5)
        blur_kernel_layout.addWidget(self.blur_kernel_slider)
        layout.addLayout(blur_kernel_layout)

        # 高斯滤波
        self.gaussian_checkbox = QCheckBox("启用高斯滤波")
        layout.addWidget(self.gaussian_checkbox)

        gaussian_kernel_layout = QVBoxLayout()
        gaussian_kernel_layout.addWidget(QLabel("卷积核大小："))
        self.gaussian_kernel_slider = QSlider(Qt.Horizontal)
        self.gaussian_kernel_slider.setRange(1, 30)
        self.gaussian_kernel_slider.setValue(5)
        gaussian_kernel_layout.addWidget(self.gaussian_kernel_slider)
        layout.addLayout(gaussian_kernel_layout)

        # 中值滤波
        self.median_checkbox = QCheckBox("启用中值滤波")
        layout.addWidget(self.median_checkbox)

        median_kernel_layout = QVBoxLayout()
        median_kernel_layout.addWidget(QLabel("卷积核大小："))
        self.median_kernel_slider = QSlider(Qt.Horizontal)
        self.median_kernel_slider.setRange(1, 30)
        self.median_kernel_slider.setValue(5)
        median_kernel_layout.addWidget(self.median_kernel_slider)
        layout.addLayout(median_kernel_layout)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def get_config(self):
        """获取当前所有配置参数"""
        gc.cf.load_conf()
        gc.cf.config = {
            'background': {
                'enabled': self.backgroun_enable_checkbox.isChecked(),
                'algorithms': self.alg_combox.currentText(),
                'history': self.history_slider.value(),
                'varThreshold': self.varThreshold_slider.value(),
                'detectShadows': self.detectShadows_checkbox.isChecked()
            },
            # 几何变换
            'resize': {
                'enabled': self.resize_checkbox.isChecked(),
                'width': self.width_spinbox.value(),
                'height': self.height_spinbox.value()
            },
            # 颜色空间转换
            'color_convert': {
                'enabled': self.color_convert_checkbox.isChecked(),
                'mode': self.color_mode_combo.currentText()
            },
            # 形态学操作
            'open': {
                'enabled': self.open_checkbox.isChecked(),
                'kernel_size': self.open_kernel_slider.value(),
                'iterations': self.open_iter_spinbox.value()
            },
            'close': {
                'enabled': self.close_checkbox.isChecked(),
                'kernel_size': self.close_kernel_slider.value(),
                'iterations': self.close_iter_spinbox.value()
            },
            'erode': {
                'enabled': self.erode_checkbox.isChecked(),
                'kernel_size': self.erode_kernel_slider.value(),
                'iterations': self.erode_iter_spinbox.value()
            },
            'dilate': {
                'enabled': self.dilate_checkbox.isChecked(),
                'kernel_size': self.dilate_kernel_slider.value(),
                'iterations': self.dilate_iter_spinbox.value()
            },
            'binary': {
                'enabled': self.binary_checkbox.isChecked(),
                'threshold': self.binary_threshold_slider.value()
            },
            'canny': {
                'enabled': self.canny_checkbox.isChecked(),
                'threshold1': self.canny_threshold1_slider.value(),
                'threshold2': self.canny_threshold2_slider.value()
            },
            'contour': {
                'enabled': self.contour_checkbox.isChecked(),
                'min_area': self.min_area_spinbox.value(),
                'max_area': self.max_area_spinbox.value(),
                'mode': self.contour_mode_combo.currentText(),
                'method': self.contour_method_combo.currentText(),
            },
            # 图像滤波
            'blur': {
                'enabled': self.blur_checkbox.isChecked(),
                'kernel_size': self.blur_kernel_slider.value()
            },
            'gaussian': {
                'enabled': self.gaussian_checkbox.isChecked(),
                'kernel_size': self.gaussian_kernel_slider.value()
            },
            'median': {
                'enabled': self.median_checkbox.isChecked(),
                'kernel_size': self.median_kernel_slider.value()
            }
        }
        return gc.cf.config

    def set_config(self):
        """根据提供的配置设置UI控件"""
        config = gc.cf.config

        if 'background' in config:
            self.backgroun_enable_checkbox.setChecked(config['background']['enabled'])
            self.alg_combox.setCurrentText(config['background']['algorithms'])
            self.history_slider.setValue(config['background']['history'])
            self.varThreshold_slider.setValue(config['background']['varThreshold'])
            self.detectShadows_checkbox.setChecked(config['background']['detectShadows'])

        # 几何变换
        if 'resize' in config:
            self.resize_checkbox.setChecked(config['resize']['enabled'])
            self.width_spinbox.setValue(config['resize']['width'])
            self.height_spinbox.setValue(config['resize']['height'])

        # 颜色空间转换
        if 'color_convert' in config:
            self.color_convert_checkbox.setChecked(config['color_convert']['enabled'])
            index = self.color_mode_combo.findText(config['color_convert']['mode'])
            if index >= 0:
                self.color_mode_combo.setCurrentIndex(index)

        if 'open' in config:
            self.open_checkbox.setChecked(config['open']['enabled'])
            self.open_kernel_slider.setValue(config['open']['kernel_size'])
            self.open_iter_spinbox.setValue(config['open']['iterations'])

        if 'close' in config:
            self.close_checkbox.setChecked(config['close']['enabled'])
            self.close_kernel_slider.setValue(config['close']['kernel_size'])
            self.close_iter_spinbox.setValue(config['close']['iterations'])

        # 形态学操作
        if 'erode' in config:
            self.erode_checkbox.setChecked(config['erode']['enabled'])
            self.erode_kernel_slider.setValue(config['erode']['kernel_size'])
            self.erode_iter_spinbox.setValue(config['erode']['iterations'])

        if 'dilate' in config:
            self.dilate_checkbox.setChecked(config['dilate']['enabled'])
            self.dilate_kernel_slider.setValue(config['dilate']['kernel_size'])
            self.dilate_iter_spinbox.setValue(config['dilate']['iterations'])

        if 'binary' in config:
            self.binary_checkbox.setChecked(config['binary']['enabled'])
            self.binary_threshold_slider.setValue(config['binary']['threshold'])

        if 'canny' in config:
            self.canny_checkbox.setChecked(config['canny']['enabled'])
            self.canny_threshold1_slider.setValue(config['canny']['threshold1'])
            self.canny_threshold2_slider.setValue(config['canny']['threshold2'])

        if 'contour' in config:
            self.contour_checkbox.setChecked(config['contour']['enabled'])
            self.min_area_spinbox.setValue(config['contour']['min_area'])
            self.max_area_spinbox.setValue(config['contour']['max_area'])
            self.contour_mode_combo.setCurrentText(config['contour']['mode'])
            self.contour_method_combo.setCurrentText(config['contour']['method'])

        # 图像滤波
        if 'blur' in config:
            self.blur_checkbox.setChecked(config['blur']['enabled'])
            self.blur_kernel_slider.setValue(config['blur']['kernel_size'])

        if 'gaussian' in config:
            self.gaussian_checkbox.setChecked(config['gaussian']['enabled'])
            self.gaussian_kernel_slider.setValue(config['gaussian']['kernel_size'])

        if 'median' in config:
            self.median_checkbox.setChecked(config['median']['enabled'])
            self.median_kernel_slider.setValue(config['median']['kernel_size'])
        logger.info("配置文件加载成功")

    def close(self):
        gc.cf.save_conf()

    def background_group(self):
        group = QGroupBox("背景处理")
        vbox = QVBoxLayout()
        group.setLayout(vbox)

        self.backgroun_enable_checkbox = QCheckBox("启用背景处理")

        # 算法选择使用水平布局
        self.alg_combox = QComboBox()
        self.alg_combox.addItems(["KNN", "MOG2"])

        # 滑块设置
        self.history_slider = QSlider(Qt.Horizontal)
        self.history_slider.setRange(1, 10000)
        self.history_slider.setValue(500)

        self.varThreshold_slider = QSlider(Qt.Horizontal)
        self.varThreshold_slider.setRange(0, 30)
        self.varThreshold_slider.setValue(6)  # 初始值设为 6，范围内的值

        self.detectShadows_checkbox = QCheckBox("detectShadows")
        self.detectShadows_checkbox.setChecked(True)

        # 使用 QFormLayout 来组织参数设置
        formlayout = QFormLayout()
        formlayout.addRow("算法选择", self.alg_combox)
        formlayout.addRow("history", self.history_slider)
        formlayout.addRow("varThreshold", self.varThreshold_slider)
        formlayout.addRow("阴影检测", self.detectShadows_checkbox)

        vbox.addWidget(self.backgroun_enable_checkbox)
        vbox.addLayout(formlayout)

        self.main_layout.addWidget(group)
