import multiprocessing
import socket
import struct
import sys
import threading
import time

from PySide6.QtCore import QObject, Signal, QThread, QTimer, Slot, QCoreApplication

from ColoredFormatter import logger as logging


class TCPConnectionWorker(QThread):
    # 当新消息到达时，发送 (connection_id, message)
    newMessage = Signal(str, object)
    # 当连接关闭时，发送 connection_id
    connectionClosed = Signal(str)

    def __init__(self, connection_id, host, port, heartbeat_interval=30, parent=None):
        super().__init__(parent)
        self.connection_id = connection_id
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.running = True
        self.sock = None
        self.lock = threading.Lock()  # 用于发送队列的线程安全
        self.send_queue = []  # 待发送数据队列
        self.buffer = b""  # 接收缓冲区

    def run(self):
        # 尝试建立连接（重试机制）
        while self.running:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=5)
                logging.info(f"[{self.connection_id}] Connected to {self.host}:{self.port}")
                self.sock.setblocking(False)
                break
            except Exception as e:
                logging.error(f"[{self.connection_id}] Connection failed: {e}")
                time.sleep(5)  # 连接失败后延时重试

        # 在当前线程创建 QTimer 用于心跳
        timer = QTimer()
        timer.timeout.connect(self.send_heartbeat)
        timer.start(self.heartbeat_interval * 1000)

        # 将 QTimer 加入事件循环（注意 QTimer 需要在事件循环中运行）
        self._timer = timer

        while self.running:
            # 检查待发送队列（顺序发送，防止乱序）
            with self.lock:
                if self.send_queue:
                    data = self.send_queue.pop(0)
                    try:
                        # 对消息进行 utf-8 编码并加上 4 字节的长度前缀
                        length_prefix = struct.pack("!I", len(data))
                        self.sock.sendall(length_prefix + data)
                    except Exception as e:
                        logging.error(f"[{self.connection_id}] Send error: {e}")
                        self.handle_disconnect()
                        continue

            # 尝试读取数据，注意非阻塞读取
            try:
                data = self.sock.recv(4096)
                if data:
                    self.buffer += data
                    self.process_buffer()
                else:
                    # 对方关闭了连接
                    logging.warning(f"[{self.connection_id}] Remote closed connection")
                    self.handle_disconnect()
            except BlockingIOError:
                # 没有数据可读
                pass
            except Exception as e:
                logging.error(f"[{self.connection_id}] Receive error: {e}")
                self.handle_disconnect()

            self.msleep(10)  # 每次循环休眠10ms，降低CPU占用

    def process_buffer(self):
        """
        使用长度前缀方式解析完整消息，防止粘包/拆包问题
        """
        while True:
            if len(self.buffer) < 4:
                break  # 长度不够一个头部
            length = struct.unpack("!I", self.buffer[:4])[0]
            if len(self.buffer) < 4 + length:
                break  # 数据还没接收完
            message_data = self.buffer[4:4 + length]
            try:
                message = message_data.decode("utf-8")
            except Exception as e:
                logging.error(f"[{self.connection_id}] Decoding error: {e}")
                message = ""
            # 通过信号回调，将新消息传出
            self.newMessage.emit(self.connection_id, message)
            # 去除已处理数据，解决缓冲区脏数据问题
            self.buffer = self.buffer[4 + length:]

    def send(self, message: str):
        """
        将消息加入发送队列（编码为 utf-8）
        """
        data = message.encode("utf-8")
        with self.lock:
            self.send_queue.append(data)

    def send_heartbeat(self):
        """
        发送心跳消息，保持长连接
        """
        heartbeat_msg = "HEARTBEAT"
        self.send(heartbeat_msg)
        logging.debug(f"[{self.connection_id}] Heartbeat sent")

    def handle_disconnect(self):
        """
        处理断线逻辑：关闭 socket 并发出连接关闭信号
        """
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logging.error(f"[{self.connection_id}] Socket close error: {e}")
        self.connectionClosed.emit(self.connection_id)

    def disconnect(self):
        """
        优雅断开连接：关闭 socket并退出线程
        """
        self.running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
            except Exception as e:
                logging.error(f"[{self.connection_id}] Disconnect error: {e}")
        self.quit()
        self.wait()


class TCPConnectionPool(QObject):
    # 全局新消息回调信号，参数为 connection_id 和消息内容
    newMessageReceived = Signal(str, str)

    def __init__(self, max_connections=10, parent=None):
        super().__init__(parent)
        self.max_connections = max_connections
        self.connections = {}
        self.connection_counter = 0
        # 根据当前计算机核心数配置线程池（此处仅作为参考和日志输出，
        # 后续可将连接任务交给 QThreadPool 统一管理）
        self.thread_pool_size = multiprocessing.cpu_count()
        logging.info(f"Configured thread pool size: {self.thread_pool_size}")

    def add_connection(self, host, port):
        """
        添加一个新连接，超过最大连接数则拒绝
        """
        if len(self.connections) >= self.max_connections:
            logging.warning("Max connections reached, rejecting new connection.")
            return None

        connection_id = f"conn_{self.connection_counter}"
        self.connection_counter += 1
        worker = TCPConnectionWorker(connection_id, host, port)
        worker.newMessage.connect(self.handle_new_message)
        worker.connectionClosed.connect(self.handle_connection_closed)
        self.connections[connection_id] = worker
        worker.start()
        logging.info(f"Connection {connection_id} started.")
        return connection_id

    def send_message(self, connection_id, message: str):
        """
        发送消息到指定连接
        """
        if connection_id in self.connections:
            self.connections[connection_id].send(message)
        else:
            logging.error(f"Connection {connection_id} not found.")

    @Slot(str, object)
    def handle_new_message(self, connection_id, message):
        """
        内部回调，将新消息传递给全局新消息回调
        """
        logging.info(f"New message from {connection_id}: {message}")
        self.newMessageReceived.emit(connection_id, message)

    @Slot(str)
    def handle_connection_closed(self, connection_id):
        """
        处理连接关闭，移除连接池中的记录
        """
        logging.info(f"Connection closed: {connection_id}")
        if connection_id in self.connections:
            del self.connections[connection_id]

    def disconnect_all(self):
        """
        一键断开所有连接（优雅退出）
        """
        for conn in list(self.connections.keys()):
            self.disconnect(conn)

    def disconnect(self, connection_id):
        """
        优雅断开指定连接
        """
        if connection_id in self.connections:
            self.connections[connection_id].disconnect()
            del self.connections[connection_id]
            logging.info(f"Connection {connection_id} disconnected.")


# 测试和示例用法
# if __name__ == "__main__":
#     app = QCoreApplication(sys.argv)
#
#     # 创建连接池，设置最大连接数为5
#     pool = TCPConnectionPool(max_connections=5)
#     # 示例：连接到本地测试服务器 127.0.0.1:12345
#     conn_id = pool.add_connection("127.0.0.1", 12345)
#
#
#     # 全局新消息回调：任意位置均可连接此信号，处理新消息
#     def global_message_handler(conn_id, msg):
#         logging.info(f"server: {conn_id}: {msg}")
#
#
#     pool.newMessageReceived.connect(global_message_handler)
#
#
#     # 示例：3秒后发送一条测试消息
#     def send_test_message():
#         if conn_id:
#             pool.send_message(conn_id, "Hello, TCP Server!")
#
#
#     timer = QTimer()
#     timer.timeout.connect(send_test_message)
#     timer.start(3000)
#
#     sys.exit(app.exec())
