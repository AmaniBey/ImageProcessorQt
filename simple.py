import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog, QSlider, QCheckBox,
                               QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QSettings, Signal, Slot
from ultralytics import YOLO


class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.yolo_model = None

    def load_yolo_model(self, model_path='yolov8n.pt'):
        """Load the YOLO model"""
        try:
            self.yolo_model = YOLO(model_path)
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False

    def load_image(self, image_path):
        """Load an image from the given path"""
        try:
            self.original_image = cv2.imread(image_path)
            self.processed_image = self.original_image.copy()
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def resize_image(self, width, height):
        """Resize the image to the given dimensions"""
        if self.original_image is None:
            return None
        try:
            self.processed_image = cv2.resize(self.original_image, (width, height))
            return self.processed_image
        except Exception as e:
            print(f"Error resizing image: {e}")
            return None

    def convert_color(self, conversion_code):
        """Convert the image to a different color space"""
        if self.processed_image is None:
            return None
        try:
            self.processed_image = cv2.cvtColor(self.processed_image, conversion_code)
            return self.processed_image
        except Exception as e:
            print(f"Error converting color: {e}")
            return None

    def erode(self, kernel_size, iterations=3):
        """Apply erosion to the image"""
        if self.processed_image is None:
            return None
        try:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.processed_image = cv2.erode(self.processed_image, kernel, iterations=iterations)
            return self.processed_image
        except Exception as e:
            print(f"Error applying erosion: {e}")
            return None

    def dilate(self, kernel_size, iterations=3):
        """Apply dilation to the image"""
        if self.processed_image is None:
            return None
        try:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.processed_image = cv2.dilate(self.processed_image, kernel, iterations=iterations)
            return self.processed_image
        except Exception as e:
            print(f"Error applying dilation: {e}")
            return None

    def binary_threshold(self, threshold_value):
        """Apply binary thresholding to the image"""
        if self.processed_image is None:
            return None
        try:
            gray = self.processed_image
            if len(self.processed_image.shape) > 2:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            _, self.processed_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            return self.processed_image
        except Exception as e:
            print(f"Error applying binary threshold: {e}")
            return None

    def canny_edge_detection(self, threshold1, threshold2):
        """Apply Canny edge detection to the image"""
        if self.processed_image is None:
            return None
        try:
            gray = self.processed_image
            if len(self.processed_image.shape) > 2:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.Canny(gray, threshold1, threshold2)
            return self.processed_image
        except Exception as e:
            print(f"Error applying Canny edge detection: {e}")
            return None

    def find_contours(self, min_area=0, max_area=float('inf')):
        """Find contours in the image"""
        if self.processed_image is None:
            return None, None
        try:
            # Ensure image is binary
            if len(self.processed_image.shape) > 2:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = self.processed_image

            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area
            filtered_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]

            # Draw contours on a copy of the original image
            contour_image = self.original_image.copy()
            cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
            self.processed_image = contour_image

            return filtered_contours, hierarchy
        except Exception as e:
            print(f"Error finding contours: {e}")
            return None, None

    def blur(self, kernel_size):
        """Apply blur to the image"""
        if self.processed_image is None:
            return None
        try:
            self.processed_image = cv2.blur(self.processed_image, (kernel_size, kernel_size))
            return self.processed_image
        except Exception as e:
            print(f"Error applying blur: {e}")
            return None

    def gaussian_blur(self, kernel_size):
        """Apply Gaussian blur to the image"""
        if self.processed_image is None:
            return None
        try:
            # Kernel size needs to be odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.processed_image = cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), 0)
            return self.processed_image
        except Exception as e:
            print(f"Error applying Gaussian blur: {e}")
            return None

    def median_blur(self, kernel_size):
        """Apply median blur to the image"""
        if self.processed_image is None:
            return None
        try:
            # Kernel size needs to be odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.processed_image = cv2.medianBlur(self.processed_image, kernel_size)
            return self.processed_image
        except Exception as e:
            print(f"Error applying median blur: {e}")
            return None

    def yolo_detect(self):
        """Run YOLO object detection on the image"""
        if self.original_image is None or self.yolo_model is None:
            return None
        try:
            results = self.yolo_model(self.original_image)
            annotated_img = results[0].plot()
            self.processed_image = annotated_img
            return annotated_img
        except Exception as e:
            print(f"Error running YOLO detection: {e}")
            return None

    def reset_processed_image(self):
        """Reset the processed image to the original"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            return True
        return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Tool")
        self.resize(1200, 800)

        self.processor = ImageProcessor()
        self.config_file = None

        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left panel - controls
        left_panel = QScrollArea()
        left_panel.setWidgetResizable(True)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # cam loading
        load_cam_label = QLabel("Load Camera")
        select_cam_combox = QComboBox()
        select_cam_combox.addItems(["0", "1", "2", "3"])
        load_cam_button = QPushButton("Load Camera")
        hbox = QHBoxLayout()
        hbox.addWidget(load_cam_label)
        hbox.addWidget(select_cam_combox)
        hbox.addWidget(load_cam_button)
        left_layout.addLayout(hbox)


        # Image loading section
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        left_layout.addWidget(load_button)

        yolo_checkbox = QCheckBox("Use YOLO8")
        left_layout.addWidget(yolo_checkbox)

        # Configuration loading/saving
        config_layout = QHBoxLayout()
        load_config_button = QPushButton("Load Config")
        load_config_button.clicked.connect(self.load_config)
        save_config_button = QPushButton("Save Config")
        save_config_button.clicked.connect(self.save_config)
        config_layout.addWidget(load_config_button)
        config_layout.addWidget(save_config_button)
        left_layout.addLayout(config_layout)

        left_layout.addWidget(QLabel("<b>Image Resizing</b>"))
        # Resize controls
        resize_check = QCheckBox("Enable Resize")
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        width_spin = QSpinBox()
        width_spin.setRange(1, 10000)
        width_spin.setValue(640)
        width_layout.addWidget(width_spin)

        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        height_spin = QSpinBox()
        height_spin.setRange(1, 10000)
        height_spin.setValue(480)
        height_layout.addWidget(height_spin)

        left_layout.addWidget(resize_check)
        left_layout.addLayout(width_layout)
        left_layout.addLayout(height_layout)

        # Color conversion
        left_layout.addWidget(QLabel("<b>Color Space Conversion</b>"))
        color_check = QCheckBox("Enable Color Conversion")
        color_combo = QComboBox()
        color_combo.addItems(["BGR2RGB", "BGR2GRAY", "BGR2HSV", "RGB2BGR", "RGB2GRAY", "RGB2HSV"])

        left_layout.addWidget(color_check)
        left_layout.addWidget(color_combo)

        # Morphological operations
        left_layout.addWidget(QLabel("<b>Morphological Operations</b>"))

        # Erosion
        erosion_check = QCheckBox("Enable Erosion")
        erosion_kernel_layout = QHBoxLayout()
        erosion_kernel_layout.addWidget(QLabel("Kernel Size:"))
        erosion_kernel_slider = QSlider(Qt.Horizontal)
        erosion_kernel_slider.setRange(1, 20)
        erosion_kernel_slider.setValue(5)
        erosion_kernel_layout.addWidget(erosion_kernel_slider)

        erosion_iter_layout = QHBoxLayout()
        erosion_iter_layout.addWidget(QLabel("Iterations:"))
        erosion_iter_spin = QSpinBox()
        erosion_iter_spin.setRange(1, 10)
        erosion_iter_spin.setValue(3)
        erosion_iter_layout.addWidget(erosion_iter_spin)

        left_layout.addWidget(erosion_check)
        left_layout.addLayout(erosion_kernel_layout)
        left_layout.addLayout(erosion_iter_layout)

        # Dilation
        dilation_check = QCheckBox("Enable Dilation")
        dilation_kernel_layout = QHBoxLayout()
        dilation_kernel_layout.addWidget(QLabel("Kernel Size:"))
        dilation_kernel_slider = QSlider(Qt.Horizontal)
        dilation_kernel_slider.setRange(1, 20)
        dilation_kernel_slider.setValue(5)
        dilation_kernel_layout.addWidget(dilation_kernel_slider)

        dilation_iter_layout = QHBoxLayout()
        dilation_iter_layout.addWidget(QLabel("Iterations:"))
        dilation_iter_spin = QSpinBox()
        dilation_iter_spin.setRange(1, 10)
        dilation_iter_spin.setValue(3)
        dilation_iter_layout.addWidget(dilation_iter_spin)

        left_layout.addWidget(dilation_check)
        left_layout.addLayout(dilation_kernel_layout)
        left_layout.addLayout(dilation_iter_layout)

        # Binary threshold
        binary_check = QCheckBox("Enable Binary Threshold")
        binary_layout = QHBoxLayout()
        binary_layout.addWidget(QLabel("Threshold:"))
        binary_slider = QSlider(Qt.Horizontal)
        binary_slider.setRange(0, 255)
        binary_slider.setValue(127)
        binary_layout.addWidget(binary_slider)

        left_layout.addWidget(binary_check)
        left_layout.addLayout(binary_layout)

        # Canny edge detection
        canny_check = QCheckBox("Enable Canny Edge Detection")
        canny_t1_layout = QHBoxLayout()
        canny_t1_layout.addWidget(QLabel("Threshold 1:"))
        canny_t1_slider = QSlider(Qt.Horizontal)
        canny_t1_slider.setRange(0, 255)
        canny_t1_slider.setValue(100)
        canny_t1_layout.addWidget(canny_t1_slider)

        canny_t2_layout = QHBoxLayout()
        canny_t2_layout.addWidget(QLabel("Threshold 2:"))
        canny_t2_slider = QSlider(Qt.Horizontal)
        canny_t2_slider.setRange(0, 255)
        canny_t2_slider.setValue(200)
        canny_t2_layout.addWidget(canny_t2_slider)

        left_layout.addWidget(canny_check)
        left_layout.addLayout(canny_t1_layout)
        left_layout.addLayout(canny_t2_layout)

        # Image filtering
        left_layout.addWidget(QLabel("<b>Image Filtering</b>"))

        # Blur
        blur_check = QCheckBox("Enable Blur")
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Kernel Size:"))
        blur_slider = QSlider(Qt.Horizontal)
        blur_slider.setRange(1, 20)
        blur_slider.setValue(5)
        blur_layout.addWidget(blur_slider)

        left_layout.addWidget(blur_check)
        left_layout.addLayout(blur_layout)

        # Gaussian Blur
        gaussian_check = QCheckBox("Enable Gaussian Blur")
        gaussian_layout = QHBoxLayout()
        gaussian_layout.addWidget(QLabel("Kernel Size:"))
        gaussian_slider = QSlider(Qt.Horizontal)
        gaussian_slider.setRange(1, 20)
        gaussian_slider.setValue(5)
        gaussian_layout.addWidget(gaussian_slider)

        left_layout.addWidget(gaussian_check)
        left_layout.addLayout(gaussian_layout)

        # Median Blur
        median_check = QCheckBox("Enable Median Blur")
        median_layout = QHBoxLayout()
        median_layout.addWidget(QLabel("Kernel Size:"))
        median_slider = QSlider(Qt.Horizontal)
        median_slider.setRange(1, 20)
        median_slider.setValue(5)
        median_layout.addWidget(median_slider)

        left_layout.addWidget(median_check)
        left_layout.addLayout(median_layout)

        # Process button
        process_button = QPushButton("Process Image")
        process_button.clicked.connect(self.process_image)
        left_layout.addWidget(process_button)

        # Reset button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_image)
        left_layout.addWidget(reset_button)

        left_layout.addStretch()
        left_panel.setWidget(left_widget)

        # Right panel - image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Original image
        original_label = QLabel("Original Image")
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 300)
        self.original_image_label.setStyleSheet("border: 1px solid black")

        # Processed image
        processed_label = QLabel("Processed Image")
        self.processed_image_label = QLabel()
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setMinimumSize(400, 300)
        self.processed_image_label.setStyleSheet("border: 1px solid black")

        right_layout.addWidget(original_label)
        right_layout.addWidget(self.original_image_label)
        right_layout.addWidget(processed_label)
        right_layout.addWidget(self.processed_image_label)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Set the main widget
        self.setCentralWidget(main_widget)

        # Store widgets for later use
        self.resize_check = resize_check
        self.width_spin = width_spin
        self.height_spin = height_spin
        self.color_check = color_check
        self.color_combo = color_combo
        self.erosion_check = erosion_check
        self.erosion_kernel_slider = erosion_kernel_slider
        self.erosion_iter_spin = erosion_iter_spin
        self.dilation_check = dilation_check
        self.dilation_kernel_slider = dilation_kernel_slider
        self.dilation_iter_spin = dilation_iter_spin
        self.binary_check = binary_check
        self.binary_slider = binary_slider
        self.canny_check = canny_check
        self.canny_t1_slider = canny_t1_slider
        self.canny_t2_slider = canny_t2_slider
        self.blur_check = blur_check
        self.blur_slider = blur_slider
        self.gaussian_check = gaussian_check
        self.gaussian_slider = gaussian_slider
        self.median_check = median_check
        self.median_slider = median_slider
        self.yolo_checkbox = yolo_checkbox

    def load_image(self):
        """Load an image file via dialog"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)")
        if file_path:
            if self.processor.load_image(file_path):
                self.display_images()

    def display_images(self):
        """Display the original and processed images"""
        if self.processor.original_image is not None:
            # Convert original image to QPixmap
            original_rgb = cv2.cvtColor(self.processor.original_image, cv2.COLOR_BGR2RGB)
            h, w, c = original_rgb.shape
            bytes_per_line = c * w
            q_img = QImage(original_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.original_image_label.width(), self.original_image_label.height(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_image_label.setPixmap(pixmap)

        if self.processor.processed_image is not None:
            # Convert processed image to QPixmap
            if len(self.processor.processed_image.shape) == 3:
                # Color image
                processed_rgb = cv2.cvtColor(self.processor.processed_image, cv2.COLOR_BGR2RGB)
                h, w, c = processed_rgb.shape
                bytes_per_line = c * w
                q_img = QImage(processed_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                # Grayscale image
                h, w = self.processor.processed_image.shape
                q_img = QImage(self.processor.processed_image.data, w, h, w, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.processed_image_label.width(), self.processed_image_label.height(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.processed_image_label.setPixmap(pixmap)

    def process_image(self):
        """Process the image with the selected operations"""
        if self.processor.original_image is None:
            return

        # Reset the processed image to original
        self.processor.reset_processed_image()

        # YOLO processing
        if self.yolo_checkbox.isChecked():
            # Load the model if not already loaded
            if self.processor.yolo_model is None:
                self.processor.load_yolo_model()

            self.processor.yolo_detect()
            self.display_images()
            return  # Skip other processing if using YOLO

        # Resize
        if self.resize_check.isChecked():
            width = self.width_spin.value()
            height = self.height_spin.value()
            self.processor.resize_image(width, height)

        # Color conversion
        if self.color_check.isChecked():
            color_code = self.color_combo.currentText()
            conversion_map = {
                "BGR2RGB": cv2.COLOR_BGR2RGB,
                "BGR2GRAY": cv2.COLOR_BGR2GRAY,
                "BGR2HSV": cv2.COLOR_BGR2HSV,
                "RGB2BGR": cv2.COLOR_RGB2BGR,
                "RGB2GRAY": cv2.COLOR_RGB2GRAY,
                "RGB2HSV": cv2.COLOR_RGB2HSV
            }
            if color_code in conversion_map:
                self.processor.convert_color(conversion_map[color_code])

        # Erosion
        if self.erosion_check.isChecked():
            kernel_size = self.erosion_kernel_slider.value()
            iterations = self.erosion_iter_spin.value()
            self.processor.erode(kernel_size, iterations)

        # Dilation
        if self.dilation_check.isChecked():
            kernel_size = self.dilation_kernel_slider.value()
            iterations = self.dilation_iter_spin.value()
            self.processor.dilate(kernel_size, iterations)

        # Binary threshold
        if self.binary_check.isChecked():
            threshold = self.binary_slider.value()
            self.processor.binary_threshold(threshold)

        # Canny edge detection
        if self.canny_check.isChecked():
            threshold1 = self.canny_t1_slider.value()
            threshold2 = self.canny_t2_slider.value()
            self.processor.canny_edge_detection(threshold1, threshold2)

        # Blur
        if self.blur_check.isChecked():
            kernel_size = self.blur_slider.value()
            self.processor.blur(kernel_size)

        # Gaussian blur
        if self.gaussian_check.isChecked():
            kernel_size = self.gaussian_slider.value()
            self.processor.gaussian_blur(kernel_size)

        # Median blur
        if self.median_check.isChecked():
            kernel_size = self.median_slider.value()
            self.processor.median_blur(kernel_size)

        # Update display
        self.display_images()

    def reset_image(self):
        """Reset the processed image to the original"""
        self.processor.reset_processed_image()
        self.display_images()

    def save_config(self):
        """Save the current configuration to a JSON file"""
        config = {
            "resize": {
                "enabled": self.resize_check.isChecked(),
                "width": self.width_spin.value(),
                "height": self.height_spin.value()
            },
            "color_conversion": {
                "enabled": self.color_check.isChecked(),
                "type": self.color_combo.currentText()
            },
            "erosion": {
                "enabled": self.erosion_check.isChecked(),
                "kernel_size": self.erosion_kernel_slider.value(),
                "iterations": self.erosion_iter_spin.value()
            },
            "dilation": {
                "enabled": self.dilation_check.isChecked(),
                "kernel_size": self.dilation_kernel_slider.value(),
                "iterations": self.dilation_iter_spin.value()
            },
            "binary_threshold": {
                "enabled": self.binary_check.isChecked(),
                "threshold": self.binary_slider.value()
            },
            "canny": {
                "enabled": self.canny_check.isChecked(),
                "threshold1": self.canny_t1_slider.value(),
                "threshold2": self.canny_t2_slider.value()
            },
            "blur": {
                "enabled": self.blur_check.isChecked(),
                "kernel_size": self.blur_slider.value()
            },
            "gaussian_blur": {
                "enabled": self.gaussian_check.isChecked(),
                "kernel_size": self.gaussian_slider.value()
            },
            "median_blur": {
                "enabled": self.median_check.isChecked(),
                "kernel_size": self.median_slider.value()
            },
            "yolo": {
                "enabled": self.yolo_checkbox.isChecked()
            }
        }

        # Create the conf directory if it doesn't exist
        conf_dir = Path("./conf")
        conf_dir.mkdir(parents=True, exist_ok=True)

        # Generate a default filename if none is provided
        if self.config_file is None:
            # Find the next available ID
            existing_files = list(conf_dir.glob("conf_*.json"))
            if existing_files:
                ids = [int(file.stem.split('_')[1]) for file in existing_files]
                next_id = max(ids) + 1
            else:
                next_id = 1
            self.config_file = conf_dir / f"conf_{next_id}.json"

        # Save the configuration
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

        print(f"Configuration saved to {self.config_file}")

    def load_config(self):
        """Load a configuration from a JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "./conf", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)

                # Update UI with loaded configuration
                self.config_file = file_path

                # Resize
                if "resize" in config:
                    self.resize_check.setChecked(config["resize"]["enabled"])
                    self.width_spin.setValue(config["resize"]["width"])
                    self.height_spin.setValue(config["resize"]["height"])

                # Color conversion
                if "color_conversion" in config:
                    self.color_check.setChecked(config["color_conversion"]["enabled"])
                    index = self.color_combo.findText(config["color_conversion"]["type"])
                    if index >= 0:
                        self.color_combo.setCurrentIndex(index)

                # Erosion
                if "erosion" in config:
                    self.erosion_check.setChecked(config["erosion"]["enabled"])
                    self.erosion_kernel_slider.setValue(config["erosion"]["kernel_size"])
                    self.erosion_iter_spin.setValue(config["erosion"]["iterations"])

                # Dilation
                if "dilation" in config:
                    self.dilation_check.setChecked(config["dilation"]["enabled"])
                    self.dilation_kernel_slider.setValue(config["dilation"]["kernel_size"])
                    self.dilation_iter_spin.setValue(config["dilation"]["iterations"])

                # Binary threshold
                if "binary_threshold" in config:
                    self.binary_check.setChecked(config["binary_threshold"]["enabled"])
                    self.binary_slider.setValue(config["binary_threshold"]["threshold"])

                # Canny
                if "canny" in config:
                    self.canny_check.setChecked(config["canny"]["enabled"])
                    self.canny_t1_slider.setValue(config["canny"]["threshold1"])
                    self.canny_t2_slider.setValue(config["canny"]["threshold2"])

                # Blur
                if "blur" in config:
                    self.blur_check.setChecked(config["blur"]["enabled"])
                    self.blur_slider.setValue(config["blur"]["kernel_size"])

                # Gaussian blur
                if "gaussian_blur" in config:
                    self.gaussian_check.setChecked(config["gaussian_blur"]["enabled"])
                    self.gaussian_slider.setValue(config["gaussian_blur"]["kernel_size"])

                # Median blur
                if "median_blur" in config:
                    self.median_check.setChecked(config["median_blur"]["enabled"])
                    self.median_slider.setValue(config["median_blur"]["kernel_size"])

                # YOLO
                if "yolo" in config:
                    self.yolo_checkbox.setChecked(config["yolo"]["enabled"])

                print(f"Configuration loaded from {file_path}")

                # If an image is loaded, process it with the new configuration
                if self.processor.original_image is not None:
                    self.process_image()

            except Exception as e:
                print(f"Error loading configuration: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())