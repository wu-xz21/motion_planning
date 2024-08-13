# -*- coding:utf-8 -*-
"""
@Author: Xianzhi Wu
@E-Mail: wuxz@outlook.com
@File: gui.py
@Time: 2024/08/12 14:53
@Introduction: I designed an ui to help users to solve the trajectory of a robotic arm easily. You just need to execute
main.py and import the path, then you can click the button to get the result. Moreover, you could select to save the
trajectory to a csv file when the trajectory is obtained.
"""
import os
import sys
import csv
import numpy as np
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QSizePolicy, QFileDialog, QLabel, QDialog, QVBoxLayout, \
    QPushButton, QWidget, QSplitter, QHBoxLayout, QTableWidget, QDoubleSpinBox, QSpinBox, QHeaderView, QMainWindow, \
    QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from utils import pso_para

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.new_window = None
        self.resize(1024, 768)
        self.setWindowTitle('机械臂轨迹求解')
        self.setFont(QFont('微软雅黑', 12))

        # --- 布局和控件 ---#

        # 中心控件：垂直布局
        self.center = QWidget()
        self.setCentralWidget(self.center)
        self.v_box = QVBoxLayout()
        self.center.setLayout(self.v_box)

        # 添加分隔器
        self.splitter0 = QSplitter()
        self.v_box.addWidget(self.splitter0)
        self.label0 = QLabel("轨迹输入参数：")
        self.splitter0.addWidget(self.label0)

        # 顶部控件：垂直布局
        self.h_box_up = QHBoxLayout()
        self.v_box.addLayout(self.h_box_up)

        # 添加表格：表示各个点的关节信息
        self.table = QTableWidget(4,6)
        self.table.setVerticalHeaderLabels(["点1", "点2", "点3", "点4"])
        self.__HTableLabels = ["J1", "J2", "J3", "J4", "J5", "J6"]
        self.table.setHorizontalHeaderLabels(self.__HTableLabels)
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                # 在QWidget中添加QDoubleSpinBox
                spinbox = QDoubleSpinBox()
                spinbox.setDecimals(3)
                spinbox.setRange(-180, 180)
                # 设置单元格的内容为QWidget
                self.table.setCellWidget(row, col, spinbox)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.h_box_up.addWidget(self.table)

        self.h_box_up.addWidget(QWidget())


        # 添加行和删除行按钮
        self.button_widget = QWidget()
        self.h_box_up.addWidget(self.button_widget)
        self.delete_layout = QVBoxLayout()
        self.button_widget.setLayout(self.delete_layout)

        self.button_import0 = QPushButton('导入点位csv文件')
        self.delete_layout.addWidget(self.button_import0)
        self.button_import0.clicked.connect(self.open_new_window)

        self.add_row_button = QPushButton("添加点")
        self.add_row_button.clicked.connect(self.add_row)
        self.delete_layout.addWidget(self.add_row_button)
        self.remove_row_button = QPushButton("删除点")
        self.remove_row_button.clicked.connect(self.remove_row)
        self.delete_layout.addWidget(self.remove_row_button)

        self.v_box.addWidget(QWidget())

        # 添加分隔器
        self.splitter1 = QSplitter()
        self.v_box.addWidget(self.splitter1)
        self.label1 = QLabel("轨迹曲线：")
        self.splitter1.addWidget(self.label1)
        self.time_label = QLabel("最优时间：")
        self.splitter1.addWidget(self.time_label)

        # 图像控件：画出轨迹曲线
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet("border: 12px solid black;")
        self.v_box.addWidget(self.canvas)

        self.v_box.addWidget(QWidget())

        # 底部控件：计算、保存等按钮
        self.h_box_down = QHBoxLayout()
        self.v_box.addLayout(self.h_box_down)

        self.freq_layout = QHBoxLayout()
        self.h_box_down.addLayout(self.freq_layout)

        self.label_freq = QLabel()
        self.label_freq.setText("轨迹更新频率：")
        self.freq_layout.addWidget(self.label_freq)
        self.freq_box = QSpinBox()
        self.freq_box.setRange(100, 1000)
        self.freq_box.setValue(200)
        self.freq_layout.addWidget(self.freq_box)

        self.button_run = QPushButton('计算')
        self.h_box_down.addWidget(self.button_run)
        self.button_run.clicked.connect(self.solve_curve)
        self.solve_curve()

        self.button_save = QPushButton('保存为csv文件')
        self.h_box_down.addWidget(self.button_save)
        self.button_save.clicked.connect(self.save_csv)

        self.v_box.addWidget(QWidget())

        # --- 变量的定义 ---#
        self.Joint = np.zeros([self.table.rowCount(), 6])
        self.pos = []       # 存储每个关节的轨迹

    def open_new_window(self):
        # 创建一个新的子窗口
        self.new_window = QDialog(self)
        self.new_window.resize(260, 260)
        self.new_window.setWindowTitle('导入文件')
        self.new_window_layout = QVBoxLayout()
        self.new_window.setLayout(self.new_window_layout)
        set_font = QFont()
        set_font.setPointSize(14)
        # 在新窗口中添加一个标签
        self.new_label = QLabel('请按照示例文件格式\n导入点位csv文件: ')
        self.new_label.setFont(set_font)
        self.new_window_layout.addWidget(self.new_label)


        self.new_window_layout.addWidget(QWidget())

        # 在新窗口中添加按钮
        self.button_import1 = QPushButton('查看示例文件')
        self.new_window_layout.addWidget(self.button_import1)
        self.button_import1.clicked.connect(self.open_example)

        self.new_window_layout.addWidget(QWidget())

        self.button_import = QPushButton('选择文件')
        self.button_import.setFont(set_font)
        self.new_window_layout.addWidget(self.button_import)
        self.button_import.clicked.connect(self.set_table_from_excel)

        self.new_window_layout.addWidget(QWidget())

        # 显示新窗口
        self.new_window.setModal(True)
        self.new_window.show()

    def set_table_from_excel(self):
        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(None, "Open CSV file", "", "Excel files (*.csv)")
        if file_path:
            # 读取 Excel 文件
            df = pd.read_csv(file_path)

        # 获取 Excel 文件的列名
        df = df[self.__HTableLabels]

        # 设置 QTableWidget 的行数
        self.table.setRowCount(len(df))

        # 设置 QTableWidget 的行名
        self.table.setVerticalHeaderLabels([f"点{i + 1}" for i in range(len(df))]
)
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                # 在QWidget中添加QDoubleSpinBox
                spinbox = QDoubleSpinBox()
                spinbox.setDecimals(3)
                spinbox.setRange(-180, 180)
                # 设置单元格的内容为QWidget
                self.table.setCellWidget(row, col, spinbox)

        # 将 Excel 文件的内容复制到 QTableWidget
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                self.table.cellWidget(i, j).setValue(value)
        self.new_window.close()

    def open_example(self):
        file_path = "examples/data_example.csv"
        if file_path:
            os.system(f'start excel "{file_path}"')
    def add_row(self):
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        self.table.setVerticalHeaderLabels([f"点{i + 1}" for i in range(row_count + 1)])
        for col in range(self.table.columnCount()):
            spinbox = QDoubleSpinBox()
            spinbox.setDecimals(3)
            spinbox.setRange(-180, 180)
            self.table.setCellWidget(row_count, col, spinbox)

    def remove_row(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
            row_count = self.table.rowCount()
            self.table.setVerticalHeaderLabels([f"点{i + 1}" for i in range(row_count)])

    def solve_curve(self):
        self.refresh_Joint()
        f = self.freq_box.value()
        # 6轴理论的最大值
        Vm = np.array([100, 100, 150, 150, 180, 180])
        Am = np.array([360, 360, 360, 360, 360, 360])

        t,self.pos,v_all = pso_para.curve(self.Joint, Am, Vm, f)
        self.plot(t,v_all)
        self.time_label.setText(f"最优时间：{t[-1]:.3f}s")

    def plot(self, t, v_all):
        # 清空图形
        self.figure.clear()
        # 获取最大时间
        maxt = max(t)
        # 绘制曲线
        labels = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.set_xlim(0, maxt + 0.25)
        for i in range(len(self.pos)):
            ax1.plot(t, self.pos[i], label=labels[i], linewidth=1.5)
        ax1.set_ylabel('Displacement$(deg)$', fontsize=14)
        ax1.grid()

        ax2 = self.figure.add_subplot(2, 1, 2)
        ax2.set_xlim(0, maxt + 0.25)
        for i in range(len(v_all)):
            ax2.plot(t, v_all[i], label=labels[i], linewidth=1.5)
        ax2.set_xlabel('Time$(s)$', fontsize=14)
        ax2.set_ylabel('Velocity$(deg/s)$', fontsize=14)
        ax2.grid()

        # 在整个 figure 的正右方显示图例
        handles, _ = ax1.get_legend_handles_labels()
        self.figure.legend(handles, labels, loc='center left', bbox_to_anchor=(0.8, 0.5), ncol=1,
                                        fontsize=18)

        # 调整图例位置，确保它不会被图像遮挡
        self.figure.subplots_adjust(right=0.75)

        self.canvas.draw()

    def refresh_Joint(self):
        numofpoints = self.table.rowCount()
        self.Joint = np.zeros([numofpoints, 6])
        for i in range(numofpoints):
            self.Joint[i] = [self.table.cellWidget(i,j).value() for j in range(6)]

    def save_csv(self):
        # 创建保存文件对话框
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
                            self, "保存文件", "", "CSV Files (*.csv);;All Files (*)", options=options)
        labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        if file_name:
            with open(file_name, 'w',newline='')as f:
                writer = csv.writer(f)
                writer.writerow(labels)
                data = self.pos.copy()
                for row in data.T:
                    writer.writerow(row)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())