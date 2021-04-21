import os
import threading
import time
import tensorflow as tf


import cv2
import dlib
import sys


size = 64
#
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 10])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)


def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnnLayer():
    # 第一层
    W1 = weightVariable([3, 3, 3, 64])  # 卷积核大小(3,3)， 输入通道(3)， 输出通道(64)
    b1 = biasVariable([64])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3, 3, 64, 128])
    b2 = biasVariable([128])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3, 3, 128, 256])
    b3 = biasVariable([256])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)

    W4 = weightVariable([3, 3, 256, 256])
    b4 = biasVariable([256])
    conv4 = tf.nn.relu(conv2d(conv3, W4) + b4)

    pool3 = maxPool(conv4)
    drop3 = dropout(pool3, keep_prob_5)

    # 第4层
    W5 = weightVariable([3, 3, 256, 512])
    b5 = biasVariable([512])
    conv5 = tf.nn.relu(conv2d(drop3, W5) + b5)

    W6 = weightVariable([3, 3, 512, 512])
    b6 = biasVariable([512])
    conv6 = tf.nn.relu(conv2d(conv5, W6) + b6)

    pool4 = maxPool(conv6)
    drop4 = dropout(pool4, keep_prob_5)

    # 全连接层
    Wf = weightVariable([4 * 4 * 512, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop4, [-1, 4 * 4 * 512])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf1 = dropout(dense, keep_prob_75)

    # 全连接层2
    Wf2 = weightVariable([512, 512])
    bf2 = biasVariable([512])
    dense2 = tf.nn.relu(tf.matmul(dropf1, Wf2) + bf2)
    dropf2 = dropout(dense2, keep_prob_75)

    # 输出层
    Wout = weightVariable([512, 10])
    bout = biasVariable([10])
    # out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf2, Wout), bout)
    return out


output = cnnLayer()
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Session()
# saver.restore(sess, tf.train.latest_checkpoint('./model'))
saver.restore(sess, './model/vgg9.model-9500')


def is_my_face(image):
    res = sess.run(predict, feed_dict={x: [image / 255.0], keep_prob_5: 1.0, keep_prob_75: 1.0})
    if res[0] == 0:
        return 0
    elif res[0] == 1:
        return 1
    elif res[0] == 2:
        return 2
    elif res[0] == 3:
        return 3
    elif res[0] == 4:
        return 4
    elif res[0] == 5:
        return 5
    elif res[0] == 6:
        return 6
    elif res[0] == 7:
        return 7
    elif res[0] == 8:
        return 8
    elif res[0] == 9:
        return 9


def put_text(face, img, x2, x1):
    if is_my_face(face) == 0:
        cv2.putText(img, "baihaozhen", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 1:
        cv2.putText(img, "huangzirui", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 2:
        cv2.putText(img, "chutianshu", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 3:
        cv2.putText(img, "wuhuan", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 4:
        cv2.putText(img, "caizhangyi", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 5:
        cv2.putText(img, "sunyihao", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 6:
        cv2.putText(img, "fansiyuan", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 7:
        cv2.putText(img, "zhaoziyu", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 8:
        cv2.putText(img, "lijunyi", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    elif is_my_face(face) == 9:
        cv2.putText(img, "chenyangzhe", (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
# cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

import cv2
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from GUI import Ui_MainWindow
import sys
import numpy as np
from ctypes import *


class myMainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)

        self.btn_open.clicked.connect(self.open_image)   # 打开视频文件按钮
        # self.btn_play.clicked.connect(self.playVideo)       # play
        self.btn_stop.clicked.connect(self.pauseVideo)       # pause

# 视频流代码 OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#         self.ui = Ui_MainWindow
#         self.mainWnd = QMainWindow

        # 默认视频源为相机
        self.RadioButtonCam.setChecked(True)
        self.isCamera = True

        # 信号槽设置
        self.Open.clicked.connect(self.open)
        self.Close.clicked.connect(self.close)
        self.RadioButtonCam.clicked.connect(self.radioButtonCam)
        self.RadioButtonFile.clicked.connect(self.radioButtonFile)

        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()


    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        # jpg = QPixmap(imgName).scaled(self.DispalyLabel.width(), self.DispalyLabel.height())
        jpg = QPixmap(imgName)
        self.DispalyLabel.setPixmap(jpg)

    def recognize(self):
        self.player.play()

    def pauseVideo(self):
        self.player.pause()


# 视频流代码 OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    def radioButtonCam(self):
        self.isCamera = True
    def close(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()

    def radioButtonFile(self):
        self.close()
        # time.sleep(3000)
        self.isCamera = False

    def open(self):
        if not self.isCamera:
            print("please select video input")
        else:
            # 下面两种rtsp格式都是支持的
            # cap = cv2.VideoCapture("rtsp://admin:Supcon1304@172.20.1.126/main/Channels/1")
            # self.cap = cv2.VideoCapture("rtsp://admin:Supcon1304@172.20.1.126:554/h264/ch1/main/av_stream")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # 创建视频显示线程
        th = threading.Thread(target=self.Display)
        th.start()



    def Display(self):
        # self.ui.Open.setEnabled(False)
        # self.ui.Close.setEnabled(True)

        while self.cap.isOpened():
            _, img = self.cap.read()
            start_time = time.time()
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_image, 1)
            if not len(dets):
                print('Can`t get face.')
                # cv2.imshow('img', img)
                # key = cv2.waitKey(30) & 0xff
                # if key == 27:
                #     sys.exit(0)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                # 调整图片的尺寸
                face = cv2.resize(face, (size, size))
                print('Is this my face? %s' % is_my_face(face))

                cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
                cv2.putText(img, "FPS {0}".format(str(1.0 / (time.time() - start_time))), (40, 40), 3, 1, (255, 0, 255),
                            2)
                put_text(face, img, x2, x1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # cv2.imshow('image', img)
                # key = cv2.waitKey(30) & 0xff
                # if key == 27:
                #     sys.exit(0)


            # success, frame_read = self.cap.read()
            # # RGB转BGR
            # #print(np.array(frame).shape)
            # start_time=time.time()
            # frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            # #frame_rgb = frame_read
            # # frame_resized = cv2.resize(frame_rgb,
            # #                            (darknet.network_width(self.netMain),
            # #                             darknet.network_height(self.netMain)),
            # #                            interpolation=cv2.INTER_LINEAR)
            # #
            # # darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())
            # #
            # # detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)
            # # image = cvDrawBoxes(detections, frame_resized)
            # #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
          

            infrence_end_time=time.time()
            cost_time=int((infrence_end_time-start_time)*1000)
            print(cost_time)

            imge = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.DispalyLabel.setPixmap(QPixmap.fromImage(imge))

            if self.isCamera:
                cv2.waitKey(1)
            else:
                self.close
                # cv2.waitKey(int(1000 / self.frameRate))

            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.DispalyLabel.clear()
                self.Close.setEnabled(False)
                self.Open.setEnabled(True)
                #  close 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
                sess.close()
                break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui_MainWindow()
    video_gui = myMainWindow()
    video_gui.show()
    sys.exit(app.exec_())
