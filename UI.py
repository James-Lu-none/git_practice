import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_AI import MCTSPlayer
from policy_value_net import PolicyValueNet  # Keras
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout, QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QPainterPath
from PyQt5.QtCore import QTimer, Qt, QPoint, QThreadPool,QRunnable,QObject,pyqtSignal,pyqtSlot
from TrainPipeline import TrainPipeline
import math
#ui--------------------------------------------------
class playoutWindow_DrawingWidget(QWidget):
    
    def __init__(self, board_size, MainWindow_self):
        super().__init__()
        
        self.Main_self=MainWindow_self
        self.board_size = board_size
        
    def paintEvent(self, event):
        super().paintEvent(event)  # 調用父類的 paintEvent 方法
        painter = QPainter(self)
        pen = QPen(Qt.black, 1)
        for col in range(15):
            painter.setPen(pen)
            painter.drawLine(29+col*40, 24, 29+col*40, 584)

        for row in range(15):
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

class playoutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        width = 650
        height = 650
        self.resize(width, height)
        self.setWindowTitle('playout')
        self.states=np.full((15,15),-1)
        self.board_size=15
        central_widget = QWidget()
        layout = QGridLayout()
        central_widget.setLayout(layout)

        self.drawing_widget = playoutWindow_DrawingWidget(self.board_size,self)
        layout.addWidget(self.drawing_widget, 1, 0, 1, 2) 
        
        self.buttons = []
        button_size = 40
        for row in range(self.board_size):
            row_buttons = []
            for col in range(self.board_size):
                button = QPushButton(self.drawing_widget)
                button.setFixedSize(button_size, button_size)
                button.setStyleSheet("background-color: transparent; border: none;") 
                button.move((row * button_size)+10, (col * button_size)+4 )
                font = QFont()
                font.setPointSize(20)
                button.setFont(font) 
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.update_board(self.states,0)
        self.setCentralWidget(central_widget)
        self.setStyleSheet("background-color: #D2B48C;")

    def update_board(self,state,root_step):
        for i in range(15):
            for j in range(15):
                if state[i][j] != -1 and state[i][j]<=root_step:
                    self.buttons[i][j].setStyleSheet('background-color: rgba(0, 0, 255, 0.3)')
                else:
                    self.buttons[i][j].setStyleSheet('background-color: transparent')
                if state[i][j] == -1:
                    self.buttons[i][j].setText("")
                    continue
                if state[i][j]%2 == 0:
                    self.buttons[i][j].setText("⚫")
                else:
                    self.buttons[i][j].setText("⚪")
                    
class DrawingWidget(QWidget):
    
    def __init__(self, board_size, MainWindow_self):
        super().__init__()
        self.Main_self=MainWindow_self
        self.board_size = board_size
    def paintEvent(self, event):
        super().paintEvent(event)  # 調用父類的 paintEvent 方法
        painter = QPainter(self)
        pen = QPen(Qt.black, 1) 
        for col in range(15):
            painter.setPen(pen)
            painter.drawLine(29+col*40, 24, 29+col*40, 584)

        for row in range(15):
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

        painter2=QPainter(self)
        painter2.setRenderHint(QPainter.Antialiasing)  # 抗锯齿效果
          # 设置画刷颜色为红色

        painter2.setRenderHint(QPainter.Antialiasing, True)
        painter2.setPen(QPen(Qt.NoPen))
        top_n_move = sorted(range(len(self.Main_self.probs)), key=lambda i: self.Main_self.probs[i], reverse=True)[:5]
        top= np.argmax(self.Main_self.probs)
        # print()
        # for i in top_n_move:
        #     print(self.Main_self.probs[i])
        # print("max",self.Main_self.probs[top])
        for row in range(self.board_size):
            for col in range(self.board_size):
                opa=self.Main_self.probs[row*15+col]
                if(self.Main_self.is_show_predict):
                    painter2.setOpacity(math.pow(opa,0.3))
                    if(row*15+col==top):
                        painter2.setBrush(QColor(255, 255, 0))
                    elif(row*15+col in top_n_move):
                        painter2.setBrush(QColor(20, 255, 100))
                    else:
                        painter2.setBrush(QColor(20, 100, 255))
                    rect=painter2.drawRect(row*40+15, col*40+9, 30, 30)

class MainWindow(QMainWindow):
    def ai_show_clicked(self):
        self.is_show_predict=not self.is_show_predict
        self.init_ui()
    def __init__(self):
        super().__init__()
        self.timer=0
        self.playout_window = None
        self.counter = 0
        self.state=np.full((15,15),-1)
        self.setWindowTitle('Gomoku')
        self.board_size = 15
        self.init_ui()
        self.is_show_predict=True
        width = 650
        height = 650
        self.resize(width, height)
        self.probs=np.full((225),0)
        self.worker = MyWorker()
        self.worker.show_board.connect(self.show_board)
        self.worker.show_probs.connect(self.show_probs)
        self.worker.show_playout_board.connect(self.show_playout_board)

        self.thread_pool = QThreadPool()
        self.start_long_loop()
    def start_long_loop(self):
        # Create a QRunnable to wrap the long-running loop function
        long_loop_runnable = LongLoopRunnable(self.worker.long_running_loop)

        # Execute the QRunnable in the thread pool
        self.thread_pool.start(long_loop_runnable)
    def init_ui(self):
        central_widget = QWidget()
        layout = QGridLayout()
        central_widget.setLayout(layout)

        self.status_label = QLabel()
        layout.addWidget(self.status_label, 0, 0)

        buttonM = QPushButton("AI 分析")
        buttonM.clicked.connect(self.ai_show_clicked)
        buttonM.setFixedSize(50, 30)  
        layout.addWidget(buttonM, 0, 1) 

        self.drawing_widget = DrawingWidget(self.board_size,self)
        layout.addWidget(self.drawing_widget, 1, 0, 1, 2) 
        
        self.buttons = []
        button_size = 40
        for row in range(self.board_size):
            row_buttons = []
            for col in range(self.board_size):
                button = QPushButton(self.drawing_widget)
                button.setFixedSize(button_size, button_size)
                button.setStyleSheet("background-color: transparent; border: none;") 
                button.move((row * button_size)+10, (col * button_size)+4 )
                font = QFont()
                font.setPointSize(20)
                button.setFont(font) 
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.update_board()
        self.setCentralWidget(central_widget)
        self.setStyleSheet("background-color: #D2B48C;")
    @pyqtSlot(list)
    def show_board(self,states):
        # print(states)
        state=np.copy(np.reshape(states,(15,15)))
        # state=np.copy(np.fliplr(state))
        self.state=state
        self.update_board()
        
        # return;
    @pyqtSlot(list)
    def show_playout_board(self,states):
        if self.playout_window is None:
            self.playout_window = playoutWindow()
            self.playout_window.show()
        board_state=states[0]
        root_board_step=states[1]
        square_states=np.copy(np.reshape(board_state,(15,15)))
        self.playout_window.update_board(square_states,root_board_step)
        

    @pyqtSlot(list)
    def show_probs(self,probs):
        # print("show new probs")
       
        self.probs=probs
        
        self.init_ui()
        
    def update_board(self):
        for i in range(15):
            for j in range(15):
                if self.state[i][j] == -1:
                    self.buttons[i][j].setText("")
                    continue
                if self.state[i][j]%2 == 0:
                    self.buttons[i][j].setText("⚫")
                else:
                    self.buttons[i][j].setText("⚪")
    def closeEvent(self, event):
        print('window closed')

class MyWorker(QObject):
    show_probs=pyqtSignal(list)
    show_board=pyqtSignal(list)
    show_playout_board=pyqtSignal(list)
    def __init__(self):
        super().__init__()

    def long_running_loop(self):
        
        # Perform your long-running tasks here
        training_pipeline = TrainPipeline(self,'./06_10_061617')
        
        training_pipeline.run()


class LongLoopRunnable(QRunnable):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        self.func()