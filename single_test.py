import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

# 图片路径
face_path_dir = './test/'
face_path = []
for i in range(10):
    name = face_path_dir + (str(i))
    face_path.append(name)


# my_faces_path = './my_faces'
# other_faces_path = './other_faces'
size = 64



def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path , h=size, w=size):
    imgs = []
    labs = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)
    labels = []
    for lab in labs:
        if lab == face_path[0]:
            lab = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif lab == face_path[1]:
            lab = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif lab == face_path[2]:
            lab = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif lab == face_path[3]:
            lab = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif lab == face_path[4]:
            lab = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif lab == face_path[5]:
            lab = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif lab == face_path[6]:
            lab = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif lab == face_path[7]:
            lab = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif lab == face_path[8]:
            lab = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif lab == face_path[9]:
            lab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        labels.append(lab)
    imgs = np.array(imgs)
    imgs = imgs.astype('float32') / 255.0
    labs = np.array(labels)
    return imgs, labs

for path in face_path:
    readData(path)
list_imgs = []
list_labs = []
for i in range(10):
    print(i)
    imgss, labss = readData(face_path[i],size,size)
    list_imgs.append(imgss)
    list_labs.append(labss)

# readData(my_faces_path)
# readData(other_faces_path)
# 将图片数据与标签转换成数组





# # 随机划分测试集与训练集
# train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.1, random_state=random.randint(0,100))
# # 参数：图片数据的总数，图片的高、宽、通道
# train_x = train_x.reshape(train_x.shape[0], size, size, 3)
# test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数


# print('size:%s' % len(imgs))

# # 图片块，每次取100张图片
# batch_size = 100
# num_batch = len(train_x) // batch_size
# print(num_batch)

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
    conv1 = tf.nn.leaky_relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3, 3, 64, 128])
    b2 = biasVariable([128])
    conv2 = tf.nn.leaky_relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3, 3, 128, 256])
    b3 = biasVariable([256])
    conv3 = tf.nn.leaky_relu(conv2d(drop2, W3) + b3)

    W4 = weightVariable([3, 3, 256, 256])
    b4 = biasVariable([256])
    conv4 = tf.nn.leaky_relu(conv2d(conv3, W4) + b4)

    pool3 = maxPool(conv4)
    drop3 = dropout(pool3, keep_prob_5)

    # 第4层
    W5 = weightVariable([3, 3, 256, 512])
    b5 = biasVariable([512])
    conv5 = tf.nn.leaky_relu(conv2d(drop3, W5) + b5)

    W6 = weightVariable([3, 3, 512, 512])
    b6 = biasVariable([512])
    conv6 = tf.nn.leaky_relu(conv2d(conv5, W6) + b6)

    pool4 = maxPool(conv6)
    drop4 = dropout(pool4, keep_prob_5)

    # 第5层
    W7 = weightVariable([3, 3, 512, 512])
    b7 = biasVariable([512])
    conv7 = tf.nn.leaky_relu(conv2d(drop4, W7) + b7)

    W8 = weightVariable([3, 3, 512, 512])
    b8 = biasVariable([512])
    conv8 = tf.nn.leaky_relu(conv2d(conv7, W8) + b8)

    pool5 = maxPool(conv8)
    drop5 = dropout(pool5, keep_prob_5)

    # 全连接层1
    Wf1 = get_weight([2 * 2 * 512, 512], regularizer=0.01)
    bf1 = biasVariable([512])
    drop5_flat = tf.reshape(drop5, [-1, 2 * 2 * 512])
    dense1 = tf.nn.leaky_relu(tf.matmul(drop5_flat, Wf1) + bf1)
    dropf1 = dropout(dense1, keep_prob_75)

    # 全连接层2
    Wf2 = get_weight([512, 512], regularizer=0.01)
    bf2 = biasVariable([512])
    dense2 = tf.nn.leaky_relu(tf.matmul(dropf1, Wf2) + bf2)
    dropf2 = dropout(dense2, keep_prob_75)

    # 输出层
    Wout = get_weight([512, 10], regularizer=0.01)
    bout = biasVariable([10])
    # out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf2, Wout), bout)
    return out

out = cnnLayer()

saver = tf.train.Saver()
# saver = tf.train.import_meta_graph('./model/train_faces1000_1.model-1334.meta')
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./model'))
    # saver.restore(sess, './model/vgg9.model-9500')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # acc1 = accuracy.eval({x:imgs[:1124], y_:labs[:1124], keep_prob_5:1.0, keep_prob_75:1.0})
    # acc2 = accuracy.eval({x: imgs[1125:], y_: labs[1125:], keep_prob_5: 1.0, keep_prob_75: 1.0})
    for i in range(10):
        acc = accuracy.eval({x:list_imgs[i], y_:list_labs[i], keep_prob_5:1.0, keep_prob_75:1.0})
        print(i,acc)
# print('test-set accuracy:', (acc1+acc2)/2)


