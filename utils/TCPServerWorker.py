import multiprocessing
import socket
import struct
import threading
import time

from PySide6.QtCore import QObject, Signal, QThread, QTimer, Slot

from ColoredFormatter import logger as logging


class TCPServerWorker(QThread):
    """
    每个客户端连接对应一个工作线程，用于处理接收、发送和心跳
    """
    # 当新消息到达时，发送 (connection_id, message)
    newMessage = Signal(str, object)
    # 当客户端断开连接时，发送 connection_id
    clientDisconnected = Signal(str)

    def __init__(self, client_socket, client_addr, connection_id, heartbeat_interval=30, parent=None):
        super().__init__(parent)
        self.client_socket = client_socket
        self.client_addr = client_addr
        self.connection_id = connection_id
        self.heartbeat_interval = heartbeat_interval
        self.running = True
        self.lock = threading.Lock()  # 用于发送队列的线程安全保护
        self.send_queue = []  # 待发送数据队列
        self.buffer = b""  # 接收缓冲区

    def run(self):
        logging.info(f"[{self.connection_id}] Handling client {self.client_addr}")

        # 设置客户端 socket 为非阻塞
        self.client_socket.setblocking(False)

        # 使用 QTimer 定时发送心跳消息
        timer = QTimer()
        timer.timeout.connect(self.send_heartbeat)
        timer.start(self.heartbeat_interval * 1000)
        self._timer = timer  # 防止被回收

        while self.running:
            # 处理待发送队列（顺序发送，防止乱序问题）
            with self.lock:
                if self.send_queue:
                    data = self.send_queue.pop(0)
                    try:
                        # 加上 4 字节长度前缀后发送
                        length_prefix = struct.pack("!I", len(data))
                        self.client_socket.sendall(length_prefix + data)
                    except Exception as e:
                        logging.error(f"[{self.connection_id}] Send error: {e}")
                        self.handle_disconnect()
                        continue

            # 尝试接收数据，非阻塞方式
            try:
                data = self.client_socket.recv(4096)
                if data:
                    self.buffer += data
                    self.process_buffer()
                else:
                    # 客户端关闭连接
                    logging.warning(f"[{self.connection_id}] Client closed connection")
                    self.handle_disconnect()
            except BlockingIOError:
                # 当前无数据可读
                pass
            except Exception as e:
                logging.error(f"[{self.connection_id}] Receive error: {e}")
                self.handle_disconnect()

            self.msleep(10)  # 降低 CPU 占用

    def process_buffer(self):
        """
        使用4字节消息长度前缀解析消息，处理粘包/拆包问题，并移除已处理数据防止缓冲区脏数据
        """
        while True:
            if len(self.buffer) < 4:
                break  # 不足以解析消息头
            length = struct.unpack("!I", self.buffer[:4])[0]
            if len(self.buffer) < 4 + length:
                break  # 消息未接收完整
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
        将消息编码为 UTF-8 后加入发送队列
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
        处理断线逻辑：关闭 socket 并发出客户端断开信号
        """
        if not self.running:
            return
        self.running = False
        try:
            self.client_socket.close()
        except Exception as e:
            logging.error(f"[{self.connection_id}] Socket close error: {e}")
        self.clientDisconnected.emit(self.connection_id)

    def disconnect(self):
        """
        优雅断开：关闭 socket 并退出线程
        """
        self.running = False
        try:
            self.client_socket.shutdown(socket.SHUT_RDWR)
            self.client_socket.close()
        except Exception as e:
            logging.error(f"[{self.connection_id}] Disconnect error: {e}")
        self.quit()
        self.wait()


class TCPServer(QObject):
    """
    TCP Server 工具类
      - 监听指定端口，接受客户端连接（超过最大连接数则拒绝）
      - 每个客户端连接由 TCPServerWorker 处理
      - 通过 newMessageReceived 信号向全局广播新消息，便于项目中任意位置进行回调处理
    """
    # 全局新消息回调信号：参数为 connection_id 和消息内容
    newMessageReceived = Signal(str, str)

    def __init__(self, host="0.0.0.0", port=12345, max_connections=10, heartbeat_interval=30, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        self.server_socket = None
        self.clients = {}  # 存储所有客户端的 TCPServerWorker 实例
        self.connection_counter = 0
        self.running = True

        # 根据 CPU 核心数配置线程池（此处仅输出日志，后续可扩展 QThreadPool 管理任务）
        self.thread_pool_size = multiprocessing.cpu_count()
        logging.info(f"Configured thread pool size: {self.thread_pool_size}")

        # 接受连接的线程（采用 Python 原生线程处理 accept 过程）
        self.accept_thread = threading.Thread(target=self.accept_clients, daemon=True)

    def start(self):
        """
        启动 TCP Server，开始监听并接受连接
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置地址复用，防止重启时端口被占用
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.setblocking(False)
        logging.info(f"TCP Server listening on {self.host}:{self.port}")
        self.accept_thread.start()

    def accept_clients(self):
        """
        接受客户端连接：
         - 超过最大连接数时拒绝连接
         - 否则创建 TCPServerWorker 处理新客户端
        """
        while self.running:
            try:
                client_socket, client_addr = self.server_socket.accept()
                logging.info(f"Incoming connection from {client_addr}")
                if len(self.clients) >= self.max_connections:
                    client_socket.send(b"Server is full, please try again later.")
                    logging.warning("Max connections reached, rejecting new connection.")
                    client_socket.close()
                    continue
                connection_id = f"client_{self.connection_counter}"
                self.connection_counter += 1
                client_socket.setblocking(False)
                worker = TCPServerWorker(client_socket, client_addr, connection_id, self.heartbeat_interval)
                worker.newMessage.connect(self.handle_new_message)
                worker.clientDisconnected.connect(self.handle_client_disconnected)
                self.clients[connection_id] = worker
                worker.start()
                logging.info(f"Client {connection_id} started.")
            except BlockingIOError:
                pass
            except Exception as e:
                logging.error(f"Accept error: {e}")
            time.sleep(0.01)

    @Slot(str, object)
    def handle_new_message(self, connection_id, message):
        """
        内部回调，将新消息通过全局信号传出
        """
        logging.info(f"New message from {connection_id}: {message}")
        self.newMessageReceived.emit(connection_id, message)

    @Slot(str)
    def handle_client_disconnected(self, connection_id):
        """
        当客户端断开连接时，从管理字典中移除对应连接
        """
        logging.info(f"Client disconnected: {connection_id}")
        if connection_id in self.clients:
            del self.clients[connection_id]

    def send_message(self, connection_id, message: str):
        """
        向指定客户端发送消息
        """
        if connection_id in self.clients:
            self.clients[connection_id].send(message)
        else:
            logging.error(f"Client {connection_id} not found.")
    def send_message_all(self, message: str):
        """
        向所有客户端发送消息
        """
        for cid in self.clients:
            self.clients[cid].send(message)

    def disconnect(self, connection_id):
        """
        优雅断开指定客户端连接
        """
        if connection_id in self.clients:
            self.clients[connection_id].disconnect()
            del self.clients[connection_id]
            logging.info(f"Client {connection_id} disconnected.")

    def disconnect_all(self):
        """
        一键断开所有客户端连接
        """
        for cid in list(self.clients.keys()):
            self.disconnect(cid)

    def stop(self):
        """
        停止服务器：
         - 停止接受新连接
         - 优雅断开所有客户端连接
         - 关闭服务器socket
        """
        self.running = False
        try:
            self.server_socket.close()
        except Exception as e:
            logging.error(f"Server socket close error: {e}")
        self.disconnect_all()
        logging.info("TCP Server stopped.")


# 测试和示例用法
# if __name__ == "__main__":
#     app = QCoreApplication(sys.argv)
#
#     # 创建 TCP Server，监听本机 12345 端口，最大连接数设为 5
#     server = TCPServer(host="127.0.0.1", port=12345, max_connections=5, heartbeat_interval=30)
#     server.start()
#
#
#     # 全局新消息回调：任意模块都可连接此信号处理新消息
#     def global_message_handler(conn_id, msg):
#         logging.info(f"client: {conn_id}: {msg}")
#
#
#     server.newMessageReceived.connect(global_message_handler)
#
#     timer = QTimer()
#     timer.timeout.connect(lambda: server.send_message_all("Hello, TCP Server!"))
#     timer.start(3000)
#
#     # # 程序退出前优雅停止服务器
#     # def cleanup():
#     #     server.stop()
#     #     app.quit()
#     #
#     # QTimer.singleShot(30000, cleanup)  # 运行30秒后退出
#
#     sys.exit(app.exec())
