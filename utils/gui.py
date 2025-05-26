import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QLineEdit, QLabel, QTabWidget, QHBoxLayout
)
from PyQt5.QtCore import QProcess

class VisualizeGTPage(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.process = None

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("输入要可视化的目录路径：")
        self.input = QLineEdit()
        self.run_button = QPushButton("开始可视化")
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)

        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.run_button)
        layout.addWidget(self.output_box)

        self.setLayout(layout)
        self.run_button.clicked.connect(self.run_commands)

    def run_commands(self):
        self.output_box.clear()
        input_dir = self.input.text().strip()

        if not input_dir:
            self.output_box.append("❌ 请输入一个有效的目录路径。")
            return

        # 删除目录
        vis_gt_dir = "pgs/vis_gt"
        try:
            if os.path.exists(vis_gt_dir):
                shutil.rmtree(vis_gt_dir)
                self.output_box.append(f"✅ 已删除 {vis_gt_dir}")
        except Exception as e:
            self.output_box.append(f"❌ 删除目录出错: {e}")
            return

        # 运行 visualize 脚本
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

        self.process.start("python", ["utils/visualize_gt.py", input_dir, "pgs/vis_gt", "--concat"])

    def handle_stdout(self):
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8")
        self.output_box.append(output)

    def handle_stderr(self):
        error = bytes(self.process.readAllStandardError()).decode("utf-8")
        self.output_box.append(f'<span style="color:red;">{error}</span>')

    def process_finished(self):
        self.output_box.append("✅ 可视化完成。")


class ExportDatasetPage(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.process = None

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("输入 LabelStudio 项目名称：")
        self.input = QLineEdit()
        self.run_button = QPushButton("开始导出")
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)

        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.run_button)
        layout.addWidget(self.output_box)

        self.setLayout(layout)
        self.run_button.clicked.connect(self.run_command)

    def run_command(self):
        self.output_box.clear()
        project_name = self.input.text().strip()

        if not project_name:
            self.output_box.append("❌ 请输入项目名称。")
            return

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)  # stdout 和 stderr 合并
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

        command = f'./utils/fast_increase_dataset_to_c4.zsh "{project_name}"'
        self.process.start("zsh", ["-c", command])

    def handle_stdout(self):
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8")
        self.output_box.append(output)

    def handle_stderr(self):
        error = bytes(self.process.readAllStandardError()).decode("utf-8")
        self.output_box.append(f'<span style="color:red;">{error}</span>')

    def process_finished(self):
        self.output_box.append("✅ 导出完成。")


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("常用命令执行器")
        self.setGeometry(200, 200, 700, 500)

        self.tabs = QTabWidget()
        self.tabs.addTab(VisualizeGTPage(), "可视化 GT")
        self.tabs.addTab(ExportDatasetPage(), "导出数据集")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
