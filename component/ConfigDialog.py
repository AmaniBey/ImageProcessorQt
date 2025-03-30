from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton


class ConfigDialog(QDialog):
    def __init__(self, parent=None, execute: callable = None):
        super().__init__(parent)
        self.setWindowTitle("保存配置")
        self.execute = execute

        layout = QVBoxLayout()

        self.label = QLabel("请输入配置:")
        layout.addWidget(self.label)

        self.config_input = QLineEdit()
        layout.addWidget(self.config_input)

        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_config)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def save_config(self):
        config_value = self.config_input.text()
        if len(config_value) <= 0:
            self.config_input.setPlaceholderText("请输入配置名称")
            return

        if self.execute:
            config_value = f"./conf/{config_value}.json"
            self.execute(config_value)

        self.accept()  # 关闭对话框
