import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout, QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QPainterPath
from PyQt5.QtCore import QTimer, Qt, QPoint
import numpy as np

class DrawingWidget(QWidget):
    def __init__(self, board_size):
        super().__init__()  
        self.board_size = board_size
        
    def paintEvent(self, event):
        super().paintEvent(event)  # 調用父類的 paintEvent 方法
        painter = QPainter(self)
        for col in range(15):
            if col == 7:
                pen = QPen(Qt.black, 1)
            else:
                pen = QPen(Qt.black, 1) 
            painter.setPen(pen)
            painter.drawLine(29+col*40, 24, 29+col*40, 584)

        for row in range(15):
            if row == 7:
                pen = QPen(Qt.black, 1)
            else:
                pen = QPen(Qt.black, 1)  
            painter.setPen(pen)
            painter.drawLine(29, 24+row*40, 589, 24+row*40)

        # 繪制黑色圆形
        pen = QPen(Qt.black)
        painter.setPen(pen)
        painter.setBrush(Qt.black)
        radius = 5
        center1 = QPoint(149, 145)
        center2 = QPoint(469, 145)
        center3 = QPoint(309, 304)
        center4 = QPoint(149, 465)
        center5 = QPoint(469, 465)
        for center in [center1,center2,center3,center4,center5]:
            painter.drawEllipse(center, radius, radius)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.state=np.full((15,15),-1)
        self.setWindowTitle('五子棋')
        self.board_size = 15  # 棋盤尺寸
        self.player = None  # 當前玩家（黑子或白子）
        self.ai_player = None  # AI玩家（黑子或白子）
        self.board = [[-1] * self.board_size for _ in range(self.board_size)]  # 棋盤狀態
        self.game_over = False  # 遊戲結束標記
        self.player_can_play = True  # 玩家是否可以点击按钮
        self.init_ui()

        width = 650
        height = 650
        self.resize(width, height)

    def init_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.status_label = QLabel()
        self.status_label.setFixedSize(200, 15)
        layout.addWidget(self.status_label)

        self.drawing_widget = DrawingWidget(self.board_size)
        layout.addWidget(self.drawing_widget)

        self.buttons = []
        button_size = 40
        for row in range(self.board_size):
            row_buttons = []
            
            for col in range(self.board_size):
                button = QPushButton(self.drawing_widget)
                button.setFixedSize(button_size, button_size)
                button.setStyleSheet("background-color: transparent; border: none;")  # 將按鈕設置為透明
                button.move((row * button_size)+10, (col * button_size)+4 )

                font = QFont()
                font.setPointSize(20)  # 設定字體大小為 20
                button.setFont(font)  # 設定字體

                row_buttons.append(button)
            self.buttons.append(row_buttons)

        self.setCentralWidget(central_widget)
        self.setStyleSheet("background-color: #D2B48C;")

    def show_board(self):
        # for j in range(15):
        #     for i in range(15):
        #         if self.state[i][j]==0: continue
        #         if self.state[i][j] == "黑":
        #             self.buttons[i][j].setText("⚫")
        #         else:
        #             self.buttons[i][j].setText("⚪")
        # return;
        for j in range(15):
            for i in range(15):
                if self.state[i][j]==0: continue
                if self.state[i][j] == 1:
                    self.buttons[i][j].setText("⚫")
                else:
                    self.buttons[i][j].setText("⚪")
        return;
        for i in range(15):
            for j in range(15):
                self.buttons[i][j].setText("")
        for i in range(15):
            for j in range(15):
                if self.state[i][j] == -1: continue
                if self.state[i][j]%2 == 0:
                    self.buttons[i][j].setText("⚫")
                else:
                    self.buttons[i][j].setText("⚪")
        return;

    def closeEvent(self, event):
        print('window closed')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print("here you go")
    p1=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]])
    p2=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] ])

    
    window.state= p1-p2
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, '白', 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '白', 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, '黑', '黑', '黑', '白', '黑', '黑', 0, 0, 0], [0, 0, 0, 0, 0, 0, '白', 0, '白', '黑', 0, '黑', '白', 0, 0], [0, 0, 0, 0, 0, 0, '黑', '黑', '黑', 0, '白', 0, '黑', '白', 0], [0, 0, 0, 0, '黑', '黑', 0, 0, '白', '白', '白', '白', 0, '白', '白'], [0, 0, 0, 0, 0, '白', '白', '黑', '黑', '白', '黑', '白', 0, '黑', 0], [0, 0, 0, '黑', '白', 0, 0, '黑', '白', '白', '白', '黑', 0, 0, '白'], [0, 0, 0, '黑', '黑', '白', 0, 0, '黑', '白', '黑', 0, '白', '黑', '黑'], [0, 0, 0, '白', '黑', 0, 0, '白', '白', '黑', '白', 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, '白', 0, '黑', '黑', '黑', '黑', 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '黑', '白', 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '白', 0, '黑'], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '白', 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    window.show_board()
    sys.exit(app.exec_())