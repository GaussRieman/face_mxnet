import os
from mxnet.gluon.model_zoo import vision
import numpy as np
import random
import d2lzh as d2l
import cv2
import mxnet as mx
from mxnet import image
from mxnet.gluon import data as gdata


# mobilenetv2 = vision.mobilenet_v2_0_5()
#
# print(mobilenetv2)

def get_batch(range, batch_size):
    batch = []
    slice = []
    random.seed(10)
    list = np.arange(range).tolist()
    while len(list)>batch_size:
        slice = random.sample(list, batch_size)
        list = [x for x in list if x not in slice]
        batch.append(slice)
    return batch

def get_batch_data(dir, batch):
    path = os.path.join(dir, 'list.txt')
    f = open(path)
    lines = f.readlines()
    length = len(lines)
    Xs , Ys = [], []
    for i in range(len(batch)):
        line = lines[batch[i]].strip().split()
        img_name = line[0]
        X = cv2.imread(img_name)
        Y = np.asarray(list(map(float, line[1:201])), dtype=np.float).reshape(-1)
        # print(X)
        # print(Y)
        Xs.append(X)
        Ys.append(Y)
    return Xs, Ys

dir = "train_data"
batch_size = 64
length = 1448

batch = get_batch(length, batch_size)
Xs, Ys = get_batch_data(dir, batch[0])
arr = np.array(Xs)

x = mx.nd.array(arr)
# print(x.shape)
# print(type(x))

transformer = gdata.vision.transforms.ToTensor()

img = image.imread('train_data/imgs/0_51_Dresses_wearingdress_51_377_0.png')
d2l.plt.imshow(img.asnumpy())
npimg = img.asnumpy()
print(npimg.shape)

ndimg = transformer(mx.nd.array(npimg))
print(ndimg.shape)