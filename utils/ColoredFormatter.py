import os
import datetime
import logging
import threading
import queue

from logging.handlers import QueueHandler, QueueListener
from logging import FileHandler  # 从 logging 模块导入

# ---------------------------
# 自定义彩色日志 Formatter：整行日志根据级别统一上色
# ---------------------------
class UniformColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',     # 蓝色
        'INFO': '\033[92m',      # 绿色
        'WARNING': '\033[93m',   # 黄色
        'ERROR': '\033[91m',     # 红色
        'CRITICAL': '\033[1;91m' # 加粗红色
    }
    RESET = '\033[0m'

    def format(self, record):
        formatted = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}{formatted}{self.RESET}"

# ---------------------------
# 自定义文件 Handler：按日期创建日志子文件夹，每个文件最多记录指定行数（50,000 行）
# ---------------------------
class DailyRotatingLineCountFileHandler(FileHandler):
    def __init__(self, max_lines=50000, encoding="utf-8", mode="a"):
        # 按日期创建子文件夹，如 logs/2025-03-24
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        self.logs_dir = os.path.join(os.getcwd(), "logs", date_str)
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        self.max_lines = max_lines
        self.current_date = datetime.date.today()
        self.file_index = 0
        self.line_count = 0
        self.mode = mode
        self.encoding = encoding
        self.base_filename = self._get_log_filename()
        super().__init__(self.base_filename, mode=self.mode, encoding=self.encoding)

    def _get_log_filename(self):
        # 日志文件命名规则：第一份文件为 log.log，后续为 log_1.log、log_2.log 等
        if self.file_index == 0:
            filename = "log.log"
        else:
            filename = f"log_{self.file_index}.log"
        return os.path.join(self.logs_dir, filename)

    def emit(self, record):
        try:
            current_date = datetime.date.today()
            # 如果日期发生变化，则创建新的日期子文件夹，并重置计数器
            if current_date != self.current_date:
                self.current_date = current_date
                date_str = self.current_date.strftime("%Y-%m-%d")
                self.logs_dir = os.path.join(os.getcwd(), "logs", date_str)
                if not os.path.exists(self.logs_dir):
                    os.makedirs(self.logs_dir)
                self.file_index = 0
                self.line_count = 0
                self.base_filename = self._get_log_filename()
                if self.stream:
                    self.stream.close()
                self.stream = self._open()
            # 每次写入前递增行计数，超过最大行数则切换到新文件
            self.line_count += 1
            if self.line_count > self.max_lines:
                self.file_index += 1
                self.line_count = 1  # 新文件第一行
                self.base_filename = self._get_log_filename()
                if self.stream:
                    self.stream.close()
                self.stream = self._open()
            super().emit(record)
        except Exception:
            self.handleError(record)

# ---------------------------
# 单例元类：确保全局只有一个日志对象
# ---------------------------
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()  # 多线程安全

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

# ---------------------------
# 全局统一日志工具类，支持异步写入日志
# ---------------------------
class GlobalLogger(metaclass=SingletonMeta):
    def __init__(self,
                 log_level=logging.DEBUG,
                 log_format='%(asctime)s\t%(levelname)s\t%(threadName)s\t%(filename)s:%(funcName)s:%(lineno)d\t\t%(message)s'):
        self.logger = logging.getLogger("GlobalLogger")
        self.logger.setLevel(log_level)
        # 防止重复添加 Handler
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 定义日志格式
        formatter = logging.Formatter(log_format)
        colored_formatter = UniformColoredFormatter(log_format)

        # 控制台 Handler（彩色输出）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colored_formatter)

        # 文件 Handler（持久化到 logs 子文件夹）
        file_handler = DailyRotatingLineCountFileHandler(max_lines=50000)
        file_handler.setFormatter(formatter)

        # 异步队列，用于异步写入日志，避免阻塞业务线程
        self.log_queue = queue.Queue(-1)  # 无限大小队列
        queue_handler = QueueHandler(self.log_queue)
        self.logger.addHandler(queue_handler)

        # QueueListener 负责异步监听队列，将日志消息传递给目标 Handler（控制台和文件）
        self.queue_listener = QueueListener(self.log_queue, console_handler, file_handler, respect_handler_level=True)
        self.queue_listener.start()

    def get_logger(self):
        return self.logger

    def shutdown(self):
        """关闭日志系统，建议在程序退出时调用"""
        self.queue_listener.stop()
        logging.shutdown()

logger = GlobalLogger().get_logger()

# ---------------------------
# 示例使用：多线程业务代码中打印日志
# ---------------------------
# if __name__ == "__main__":
#     import time
#     from concurrent.futures import ThreadPoolExecutor
#
#     # 获取全局单例日志对象
#     logger = GlobalLogger().get_logger()
#
#     def business_logic(thread_id):
#         # 模拟业务处理，循环写日志
#         for i in range(50100):
#             logger.info(f"线程 {thread_id} 正在处理第 {i} 次业务")
#
#     # 使用 ThreadPoolExecutor 模拟多线程业务场景
#     start = time.time()
#     num_threads = 10
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         for i in range(num_threads):
#             executor.submit(business_logic, i)
#
#     logger.info("多线程业务日志测试完成")
#     logger.info(f"总耗时：{time.time() - start:.2f} 秒")
#     # 程序退出前关闭日志系统
#     GlobalLogger().shutdown()
