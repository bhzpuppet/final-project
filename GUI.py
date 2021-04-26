# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets, QtQuickWidgets


class Ui_MainWindow(object):


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(660, 678)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.text = QtWidgets.QLineEdit(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(10, 510, 640, 20))
        self.text.setObjectName("lab1")
        # self.text.setText("")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 500, 220, 220))
        self.widget.setObjectName("widget")

        # self.label = QtWidgets.QLabel(self.widget)
        # self.label.setGeometry(QtCore.QRect(20, 10, 71, 31))
        # self.label.setObjectName("label")

        self.lab2 = QtWidgets.QLabel(self.widget)
        self.lab2.setGeometry(QtCore.QRect(50, 108, 91, 31))
        self.lab2.setObjectName("lab1")
        self.lab2.setText("Camera input")
        self.lab3 = QtWidgets.QLabel(self.widget)
        self.lab3.setGeometry(QtCore.QRect(50, 58, 91, 31))
        self.lab3.setObjectName("lab3")
        self.lab3.setText("Image input")

        # 视频流代码 OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

        self.RadioButtonCam = QtWidgets.QRadioButton(self.widget)
        self.RadioButtonCam.setGeometry(QtCore.QRect(30, 108, 91, 31))
        self.RadioButtonCam.setObjectName("radioButtonCam")

        self.RadioButtonFile = QtWidgets.QRadioButton(self.widget)
        self.RadioButtonFile.setGeometry(QtCore.QRect(30, 58, 91, 31))
        self.RadioButtonFile.setObjectName("radioButtonFile")

        self.Open = QtWidgets.QPushButton(self.centralwidget)
        self.Open.setGeometry(QtCore.QRect(240, 608, 155, 40))
        self.Open.setObjectName("Open")
        self.Open.setText("Open")

        self.Close = QtWidgets.QPushButton(self.centralwidget)
        self.Close.setGeometry(QtCore.QRect(405, 608, 155, 40))
        self.Close.setObjectName("Close")
        self.Close.setText("Close")


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 摄像头显示的画面
        self.DisplayLabel = QtWidgets.QLabel(self.centralwidget)
        self.DisplayLabel.setGeometry(QtCore.QRect(10, 10, 650, 480))
        self.DisplayLabel.setMouseTracking(False)
        self.DisplayLabel.setText("")
        self.DisplayLabel.setObjectName("DispalyLabel")
        self.DisplayLabel.raise_()


        self.btn_open = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open.setGeometry(QtCore.QRect(240, 558, 100, 40))
        self.btn_open.setObjectName("btn_open")

        self.btn_play = QtWidgets.QPushButton(self.centralwidget)
        self.btn_play.setGeometry(QtCore.QRect(350, 558, 100, 40))
        self.btn_play.setObjectName("btn_play")

        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setGeometry(QtCore.QRect(460, 558, 100, 40))
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
        self.btn_open.setText(_translate("MainWindow", "Open"))
        self.btn_play.setText(_translate("MainWindow", "Recognize"))
        self.btn_stop.setText(_translate("MainWindow", "Clear"))
