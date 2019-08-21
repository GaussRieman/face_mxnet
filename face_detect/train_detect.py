import mxnet as mx
import mxnet.gluon.nn as nn
import d2lzh as d2l
import matplotlib.pyplot as plt
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, model_zoo
from mxnet.gluon import utils as gutils
import os
import time
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from face_align.face_align_model import MobilenetV2
from face_align.regressionDataSet import RegressionDataSet

import sys
sys.path.append("../")


TEST = 1
idx = 99





epoch_list = []
loss_list = []

#data preparation
data_foler = '/home/frank/Desktop/projects/DL/faceDetection/face_mxnet/data_process/cnnface'
train_list = os.path.join(data_foler, 'trainImageList.txt')
test_list = os.path.join(data_foler, 'testImageList.txt')


train_data = RegressionDataSet(train_list, img_folder=data_foler)
test_data = RegressionDataSet(test_list, img_folder=data_foler)

train_trans = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize((224,224)),
    gdata.vision.transforms.ToTensor()
])

test_trans = gdata.vision.transforms.Compose([
gdata.vision.transforms.Resize((224,224)),
    gdata.vision.transforms.ToTensor()
])



#model
pretrained_net = model_zoo.vision.mobilenet_v2_1_0(pretrained=True)
finetune_net = model_zoo.vision.mobilenet_v2_1_0(classes=4)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Normal(sigma=0.05))

ratio = 250.0/224.0

#train function
def train(train_iter, net, loss, trainer, batch_size, num_epochs, ctx):
    """Train and evaluate a model."""
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        # print("lr = ", trainer.learning_rate)
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = d2l.utils._get_batch(batch, ctx)
            ys = [x/250.0 for x in ys]
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            m += sum([y.size for y in ys])

        epoch_list.append(epoch)
        loss_list.append(train_l_sum / n)
        print('epoch %d, loss %.8f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, 0, 0,
                 time.time() - start))


def train_fine_tuning(net, learning_rate, batch_size=32, num_epochs=5):
    train_iter = gdata.DataLoader(
        train_data.transform_first(train_trans), batch_size, shuffle=True)
    test_iter = gdata.DataLoader(
        test_data.transform_first(test_trans), batch_size)
    ctx = d2l.try_all_gpus()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    if os.path.exists('face_cnn.params'):
        print('load params from existing file')
        net.load_parameters('face_cnn.params')
    # lrs = mx.lr_scheduler.FactorScheduler(500, 0.8)
    loss = gloss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    train(train_iter, net, loss, trainer, batch_size, num_epochs, ctx)


if TEST:
    trans = gdata.vision.transforms.ToTensor()
    finetune_net.load_parameters('face_cnn.params')
    # train_file = '../data_process/cnnface/lfw_5590/Aaron_Eckhart_0001.jpg'
    f = open('../data_process/cnnface/trainImageList.txt')
    lines = f.readlines()
    line = lines[idx]
    line = line.strip().split()
    label = line[1:]
    label = [int(float(x)) for x in label]
    print(label)

    path = os.path.join(data_foler, line[0])
    # print(path)
    img = cv2.imread(path)
    ratio_x = img.shape[1]/224.0
    ratio_y = img.shape[0]/224.0
    # print(img.shape)
    img = cv2.resize(img, (224,224))
    t_img = trans(mx.nd.array(img))
    t_img = mx.ndarray.expand_dims(t_img, axis = 0)


    out = finetune_net(t_img)
    out_label = out[0]
    array = out_label.asnumpy()
    # print(array)
    array = [int(x*224) for x in array]
    print(array)


    # print(ratio_x)
    # array = [int(x/ratio_x) for x in array]
    label = [int(x/ratio_x) for x in label]
    print(label)

    loss = 0.0
    for i in range(len(array)):
        loss += (label[i] - array[i])**2
        print('gt:', label[i], 'pre:', array[i])
    print(loss/10.0)

    cv2.rectangle(img, (label[0], label[2]), (label[1], label[3]), (255,0,0))
    cv2.rectangle(img, (array[0], array[2]), (array[1], array[3]), (0,255,0))
    # for i in range(5):
    #     cv2.circle(img, (array[4+2*i], array[4+2*i+1]), 2, (0,255,0), 1)
    #     cv2.circle(img, (label[2*i], label[2*i+1]), 2, (255,0, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
else:
    train_fine_tuning(finetune_net, 0.008, batch_size=32, num_epochs=20)
    finetune_net.save_parameters('face_cnn.params')
    plt.figure()
    plt.plot(epoch_list, loss_list)
    plt.show()



