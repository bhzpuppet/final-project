# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets, QtQuickWidgets


class Ui_MainWindow(object):


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1010, 980)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.lab1 = QtWidgets.QLabel(self.centralwidget)
        self.lab1.setGeometry(QtCore.QRect(10, 420, 91, 31))
        self.lab1.setObjectName("lab1")
        self.lab1.setText("模式选择")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 400, 220, 220))
        self.widget.setObjectName("widget")

        # self.label = QtWidgets.QLabel(self.widget)
        # self.label.setGeometry(QtCore.QRect(20, 10, 71, 31))
        # self.label.setObjectName("label")

        self.lab2 = QtWidgets.QLabel(self.widget)
        self.lab2.setGeometry(QtCore.QRect(50, 58, 91, 31))
        self.lab2.setObjectName("lab1")
        self.lab2.setText("摄像头输入")
        self.lab3 = QtWidgets.QLabel(self.widget)
        self.lab3.setGeometry(QtCore.QRect(50, 98, 91, 31))
        self.lab3.setObjectName("lab1")
        self.lab3.setText("视频文件输入")

        # 视频流代码 OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

        self.RadioButtonCam = QtWidgets.QRadioButton(self.widget)
        self.RadioButtonCam.setGeometry(QtCore.QRect(30, 58, 91, 31))
        self.RadioButtonCam.setObjectName("radioButtonCam")

        self.RadioButtonFile = QtWidgets.QRadioButton(self.widget)
        self.RadioButtonFile.setGeometry(QtCore.QRect(30, 98, 91, 31))
        self.RadioButtonFile.setObjectName("radioButtonFile")

        self.Open = QtWidgets.QPushButton(self.centralwidget)
        self.Open.setGeometry(QtCore.QRect(350, 500, 100, 41))
        self.Open.setObjectName("Open")
        self.Open.setText("打开")

        self.Close = QtWidgets.QPushButton(self.centralwidget)
        self.Close.setGeometry(QtCore.QRect(550, 500, 100, 41))
        self.Close.setObjectName("Close")
        self.Close.setText("关闭")


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 摄像头显示的画面
        self.DispalyLabel = QtWidgets.QLabel(self.centralwidget)
        self.DispalyLabel.setGeometry(QtCore.QRect(200, 0, 711, 411))
        self.DispalyLabel.setMouseTracking(False)
        self.DispalyLabel.setText("")
        self.DispalyLabel.setObjectName("DispalyLabel")
        self.DispalyLabel.raise_()


        self.btn_open = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open.setGeometry(QtCore.QRect(260, 440, 100, 51))
        self.btn_open.setObjectName("btn_open")

        self.btn_play = QtWidgets.QPushButton(self.centralwidget)
        self.btn_play.setGeometry(QtCore.QRect(420, 440, 71, 51))
        self.btn_play.setObjectName("btn_play")

        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setGeometry(QtCore.QRect(560, 440, 71, 51))
        self.btn_stop.setObjectName("btn_stop")

        # self.lab_video = QtWidgets.QLabel(self.centralwidget)
        # self.lab_video.setGeometry(QtCore.QRect(680, 380, 91, 31))
        # self.lab_video.setObjectName("lab_video")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 568, 23))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_open.setText(_translate("MainWindow", "打开"))
        self.btn_play.setText(_translate("MainWindow", "播放"))
        self.btn_stop.setText(_translate("MainWindow", "暂停"))
