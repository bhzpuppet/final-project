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

# my_faces_path = './my_faces'
# other_faces_path = './other_faces'
size = 64
#
# imgs = []
# labs = []
#
# def getPaddingSize(img):
#     h, w, _ = img.shape
#     top, bottom, left, right = (0,0,0,0)
#     longest = max(h, w)
#
#     if w < longest:
#         tmp = longest - w
#         # //表示整除符号
#         left = tmp // 2
#         right = tmp - left
#     elif h < longest:
#         tmp = longest - h
#         top = tmp // 2
#         bottom = tmp - top
#     else:
#         pass
#     return top, bottom, left, right
#
# def readData(path , h=size, w=size):
#     for filename in os.listdir(path):
#         if filename.endswith('.jpg'):
#             filename = path + '/' + filename
#
#             img = cv2.imread(filename)
#
#             top,bottom,left,right = getPaddingSize(img)
#             # 将图片放大， 扩充图片边缘部分
#             img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
#             img = cv2.resize(img, (h, w))
#
#             imgs.append(img)
#             labs.append(path)
#
# readData(my_faces_path)
# readData(other_faces_path)
# # 将图片数据与标签转换成数组
# imgs = np.array(imgs)
# labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])
# # 随机划分测试集与训练集
# train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))
# # 参数：图片数据的总数，图片的高、宽、通道
# train_x = train_x.reshape(train_x.shape[0], size, size, 3)
# test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# # 将数据转换成小于1的数
# train_x = train_x.astype('float32')/255.0
# test_x = test_x.astype('float32')/255.0
#
# print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# # 图片块，每次取128张图片
# batch_size = 128
# num_batch = len(train_x) // 128
#
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 10])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # 第一层
    W1 = weightVariable([3, 3, 3, 64])  # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
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

    # 第5层
    W7 = weightVariable([3, 3, 512, 512])
    b7 = biasVariable([512])
    conv7 = tf.nn.relu(conv2d(drop4, W7) + b7)

    W8 = weightVariable([3, 3, 512, 512])
    b8 = biasVariable([512])
    conv8 = tf.nn.relu(conv2d(conv7, W8) + b8)

    pool5 = maxPool(conv8)
    drop5 = dropout(pool5, keep_prob_5)

    # 全连接层1
    Wf1 = get_weight([2 * 2 * 512, 512], regularizer=0.0001)
    bf1 = biasVariable([512])
    drop5_flat = tf.reshape(drop5, [-1, 2 * 2 * 512])
    dense1 = tf.nn.relu(tf.matmul(drop5_flat, Wf1) + bf1)
    dropf1 = dropout(dense1, keep_prob_75)

    # 全连接层2
    Wf2 = get_weight([512, 512], regularizer=0.0001)
    bf2 = biasVariable([512])
    dense2 = tf.nn.relu(tf.matmul(dropf1, Wf2) + bf2)
    dropf2 = dropout(dense2, keep_prob_75)

    # 输出层
    Wout = get_weight([512, 10], regularizer=0.0001)
    bout = biasVariable([10])
    # out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf2, Wout), bout)
    return out

output = cnnLayer()  
predict = tf.argmax(output, 1)  
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('./model'))
# saver.restore(sess, './model/vgg9.model-9500')
   
def is_my_face(image):  
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
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


# import numpy
# from PIL import Image, ImageDraw, ImageFont
#
# def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
#     if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(img)
#     # 字体的格式
#     fontStyle = ImageFont.truetype(
#         "font/simsun.ttc", textSize, encoding="utf-8")
#     # 绘制文本
#     draw.text((left, top), text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
#     cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)






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




#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
   
while True:  
    _, img = cam.read()
    start_time = time.time()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        #print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff  
        if key == 27:
            sys.exit(0)
            
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1,x2:y2]
        # 调整图片的尺寸
        face = cv2.resize(face, (size,size))
        print('Is this my face? %s' % is_my_face(face))

        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
        cv2.putText(img, "FPS {0}".format(str(1.0 / (time.time() - start_time))), (40, 40), 3, 1, (255, 0, 255), 2)
        put_text()
        cv2.imshow('image',img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)
  
sess.close() 
