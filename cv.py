import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
import time

output_dir = './image'
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
saver.restore(sess, './model/vgg11_reg_leaky5.model-8000')


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


def put_text():
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

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
index = 1
while True:
    index = index + 1
    print(index)
    _, img = cam.read()
    start_time = time.time()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        # print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]
        # 调整图片的尺寸
        face = cv2.resize(face, (size, size))
        print('Is this my face? %s' % is_my_face(face))
        key = cv2.waitKey(30) & 0xff
        if key == 13:
            cv2.imwrite(output_dir + '/' + str(index) + '.jpg', img)
        if key == 27:
            sys.exit(0)

        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        cv2.putText(img, "FPS {0}".format(str(1.0 / (time.time() - start_time))), (40, 40), 3, 1, (255, 0, 255), 2)
        put_text()
        cv2.imshow('image', img)


sess.close()
