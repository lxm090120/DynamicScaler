import sys

class CustomLog:
    def __init__(self):
        self.terminal = sys.stdout  # 保存当前的 stdout
        self.log_file = None

    def log_to_file_and_terminal(self, log_file_path):
        if self.log_file is not None:
            self.reset_log_to_terminal()  # 先恢复原来的 stdout

        self.log_file = open(log_file_path, 'a')  # 打开文件以追加方式写入

        # 重定向 stdout
        sys.stdout = self
        print(f"sys stdout has been forwarded to both terminal and {log_file_path}")

    def write(self, message):
        self.terminal.write(message)  # 输出到控制台
        self.log_file.write(message)   # 写入文件

    def flush(self):
        pass  # 这个方法可以是空的，以支持 flush

    def reset_log_to_terminal(self):
        if self.log_file is not None:
            self.log_file.close()  # 关闭文件
            self.log_file = None
        sys.stdout = self.terminal  # 恢复到原始 stdout
        print(f"sys stdout has been reset to terminal")