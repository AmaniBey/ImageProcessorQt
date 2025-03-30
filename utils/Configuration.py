import json
import os

import cv2

from utils.ColoredFormatter import GlobalLogger

logger = GlobalLogger().get_logger()


class Configuration:
    DEFAULT_SAVE_LOAD_PATH = "./conf/conf.json"
    DEFAULT_CONFIG = {
        'background': {
            'enabled': False,
            "algorithms": "KNN",
            "history": 500,
            "varThreshold": 6,
            "detectShadows": True
        },
        # 几何变换
        'resize': {
            'enabled': False,
            'width': 640,
            'height': 360
        },
        # 颜色空间转换
        'color_convert': {
            'enabled': False,
            'mode': "BGR2RGB"
        },
        # 形态学操作
        'open': {
            'enabled': False,
            'kernel_size': 5,
            'iterations': 1
        },
        'close': {
            'enabled': False,
            'kernel_size': 5,
            'iterations': 1

        },
        'erode': {
            'enabled': False,
            'kernel_size': 5,
            'iterations': 1
        },
        'dilate': {
            'enabled': False,
            'kernel_size': 5,
            'iterations': 1
        },
        # 二值化
        'binary': {
            'enabled': False,
            'threshold': 127
        },
        # Canny 边缘检测
        'canny': {
            'enabled': False,
            'threshold1': 100,
            'threshold2': 200
        },
        # 轮廓检测
        'contour': {
            'enabled': False,
            'min_area': 103,
            'max_area': 10000,
            'mode': 'RETR_EXTERNAL',
            'method': "CHAIN_APPROX_NONE"
        },
        # 图像滤波
        'blur': {
            'enabled': False,
            'kernel_size': 7
        },
        'gaussian': {
            'enabled': False,
            'kernel_size': 21,
            'sigma': 1.0
        },
        'median': {
            'enabled': False,
            'kernel_size': 5
        }
    }

    COLOR_MAP = {
        "BGR2GRAY": cv2.COLOR_BGR2GRAY,
        "BGR2RGB": cv2.COLOR_BGR2RGB,
        "BGR2HSV": cv2.COLOR_BGR2HSV,
        "RGB2BGR": cv2.COLOR_RGB2BGR,
        "RGB2GRAY": cv2.COLOR_RGB2GRAY,
        "RGB2HSV": cv2.COLOR_RGB2HSV,
    }
    CONTOUR_MODE_MAP = {
        "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
        "RETR_LIST": cv2.RETR_LIST,
        "RETR_CCOMP": cv2.RETR_CCOMP,
        "RETR_TREE": cv2.RETR_TREE,
    }
    CONTOUR_METHOD_MAP = {
        "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
        "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
    }

    def __init__(self, config_file=None):
        self.config_file = config_file or self.DEFAULT_SAVE_LOAD_PATH
        self.config = self.DEFAULT_CONFIG.copy()
        self.knn: cv2.BackgroundSubtractorKNN = cv2.createBackgroundSubtractorKNN()
        self.mog2: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2()
        self.load_conf()

    def load_conf(self):
        """Load configuration from a JSON file or create if not exists."""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                try:
                    self.config.update(json.load(f))
                except json.JSONDecodeError:
                    logger.error(f"Error loading configuration from {self.config_file}. Using default values.")
        else:
            logger.info(f"Configuration file {self.config_file} not found. Creating a new one with default values.")
            self.save_conf()  # Create the config file with default values

    def save_conf(self):
        """Save the current configuration to a JSON file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)  # Create the folder if it doesn't exist
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_file}: {e}")
    def save_conf_by_path(self,path):
        """Save the current configuration to a JSON file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)  # Create the folder if it doesn't exist
            with open(path, "w") as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_file}: {e}")

    def get_conf_info(self):
        """Return the current configuration dictionary."""
        return self.config

    def get_conf_info(self):
        """Return the current configuration dictionary."""
        return self.config

    # Getters and Setters
    def get_resize_enabled(self):
        return self.config['resize']['enabled']

    def set_resize_enabled(self, value):
        self.config['resize']['enabled'] = value

    def get_resize_width(self):
        return self.config['resize']['width']

    def set_resize_width(self, value):
        self.config['resize']['width'] = value

    def get_resize_height(self):
        return self.config['resize']['height']

    def set_resize_height(self, value):
        self.config['resize']['height'] = value

    def get_color_convert_enabled(self):
        return self.config['color_convert']['enabled']

    def set_color_convert_enabled(self, value):
        self.config['color_convert']['enabled'] = value

    def get_color_convert_mode(self):
        return self.COLOR_MAP.get(self.config['color_convert']['mode'], cv2.COLOR_RGB2GRAY)

    def set_color_convert_mode(self, value):
        self.config['color_convert']['mode'] = value

    def get_erode_enabled(self):
        return self.config['erode']['enabled']

    def set_erode_enabled(self, value):
        self.config['erode']['enabled'] = value

    def get_erode_kernel_size(self):
        return self.config['erode']['kernel_size']

    def set_erode_kernel_size(self, value):
        self.config['erode']['kernel_size'] = value

    def get_erode_iterations(self):
        return self.config['erode']['iterations']

    def set_erode_iterations(self, value):
        self.config['erode']['iterations'] = value

    def get_dilate_enabled(self):
        return self.config['dilate']['enabled']

    def set_dilate_enabled(self, value):
        self.config['dilate']['enabled'] = value

    def get_dilate_kernel_size(self):
        return self.config['dilate']['kernel_size']

    def set_dilate_kernel_size(self, value):
        self.config['dilate']['kernel_size'] = value

    def get_dilate_iterations(self):
        return self.config['dilate']['iterations']

    def set_dilate_iterations(self, value):
        self.config['dilate']['iterations'] = value

    def get_binary_enabled(self):
        return self.config['binary']['enabled']

    def set_binary_enabled(self, value):
        self.config['binary']['enabled'] = value

    def get_binary_threshold(self):
        return self.config['binary']['threshold']

    def set_binary_threshold(self, value):
        self.config['binary']['threshold'] = value

    def get_canny_enabled(self):
        return self.config['canny']['enabled']

    def set_canny_enabled(self, value):
        self.config['canny']['enabled'] = value

    def get_canny_threshold1(self):
        return self.config['canny']['threshold1']

    def set_canny_threshold1(self, value):
        self.config['canny']['threshold1'] = value

    def get_canny_threshold2(self):
        return self.config['canny']['threshold2']

    def set_canny_threshold2(self, value):
        self.config['canny']['threshold2'] = value

    def get_contour_enabled(self):
        return self.config['contour']['enabled']

    def set_contour_enabled(self, value):
        self.config['contour']['enabled'] = value

    def get_contour_min_area(self):
        return self.config['contour']['min_area']

    def set_contour_min_area(self, value):
        self.config['contour']['min_area'] = value

    def get_contour_max_area(self):
        return self.config['contour']['max_area']

    def set_contour_max_area(self, value):
        self.config['contour']['max_area'] = value

    def get_contour_mode(self):
        return self.CONTOUR_MODE_MAP.get(self.config['contour']['mode'], cv2.RETR_LIST)

    def set_contour_mode(self, value):
        self.config['contour']['mode'] = value

    def get_contour_method(self):
        return self.CONTOUR_METHOD_MAP.get(self.config['contour']['method'], cv2.CHAIN_APPROX_SIMPLE)

    def set_contour_method(self, value):
        self.config['contour']['method'] = value

    def get_blur_enabled(self):
        return self.config['blur']['enabled']

    def set_blur_enabled(self, value):
        self.config['blur']['enabled'] = value

    def get_blur_kernel_size(self):
        return self.config['blur']['kernel_size']

    def set_blur_kernel_size(self, value):
        self.config['blur']['kernel_size'] = value

    def get_gaussian_enabled(self):
        return self.config['gaussian']['enabled']

    def set_gaussian_enabled(self, value):
        self.config['gaussian']['enabled'] = value

    def get_gaussian_kernel_size(self):
        return self.config['gaussian']['kernel_size']

    def set_gaussian_kernel_size(self, value):
        self.config['gaussian']['kernel_size'] = value

    def get_gaussian_sigma(self):
        return self.config['gaussian']['sigma']

    def set_gaussian_sigma(self, value):
        self.config['gaussian']['sigma'] = value

    def get_median_enabled(self):
        return self.config['median']['enabled']

    def set_median_enabled(self, value):
        self.config['median']['enabled'] = value

    def get_median_kernel_size(self):
        return self.config['median']['kernel_size']

    def set_median_kernel_size(self, value):
        self.config['median']['kernel_size'] = value

    def get_background_enable(self):
        return self.config['background']['enabled']

    def get_background_algorithms(self):
        if self.config['background']['algorithms'] == 'KNN':
            self.knn.setHistory(self.config['background']['history'])
            self.knn.setDetectShadows(self.config['background']['detectShadows'])
            self.knn.setDist2Threshold(self.config['background']['varThreshold'])
            return self.knn
        else:
            self.mog2.setHistory(self.config['background']['history'])
            self.mog2.setDetectShadows(self.config['background']['detectShadows'])
            self.mog2.setVarThreshold(self.config['background']['varThreshold'])
            return self.mog2

    def get_background_varThreshold(self):
        return self.config['background']['varThreshold']

    def get_background_history(self):
        return self.config['background']['history']

    def get_background_detectShadows(self):
        return self.config['background']['detectShadows']

    def set_background_enable(self, value):
        self.config['background']['enable'] = value

    def set_background_algorithms(self, value):
        self.config['background']['algorithms'] = value

    def set_background_varThreshold(self, value):
        self.config['background']['varThreshold'] = value

    def set_background_history(self, value):
        self.config['background']['history'] = value

    def set_background_detectShadows(self, value):
        self.config['background']['detectShadows'] = value

        # Getter and Setter for open_enabled

    def get_open_enabled(self):
        return self.config['open']['enabled']

    def set_open_enabled(self, value):
        self.config['open']['enabled'] = value

    # Getter and Setter for open_kernel_size
    def get_open_kernel_size(self):
        return self.config['open']['kernel_size']

    def set_open_kernel_size(self, value):
        self.config['open']['kernel_size'] = value

    # Getter and Setter for open_iterations
    def get_open_iterations(self):
        return self.config['open']['iterations']

    def set_open_iterations(self, value):
        self.config['open']['iterations'] = value

    # Getter and Setter for close_enabled
    def get_close_enabled(self):
        return self.config['close']['enabled']

    def set_close_enabled(self, value):
        self.config['close']['enabled'] = value

    # Getter and Setter for close_kernel_size
    def get_close_kernel_size(self):
        return self.config['close']['kernel_size']

    def set_close_kernel_size(self, value):
        self.config['close']['kernel_size'] = value

    # Getter and Setter for close_iterations
    def get_close_iterations(self):
        return self.config['close']['iterations']

    def set_close_iterations(self, value):
        self.config['close']['iterations'] = value
