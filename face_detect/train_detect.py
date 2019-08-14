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

import sys
sys.path.append("../")

from face_align.face_align_model import MobilenetV2
from face_align.regressionDataSet import RegressionDataSet



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

# print(train_list)



#model
pretrained_net = model_zoo.vision.mobilenet_v2_1_0(pretrained=True)
finetune_net = model_zoo.vision.mobilenet_v2_1_0(classes=14)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Normal(sigma=0.05))


#train function
def train(train_iter, net, loss, trainer, batch_size, num_epochs, ctx):
    """Train and evaluate a model."""
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = d2l.utils._get_batch(batch, ctx)
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
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
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
    lrs = mx.lr_scheduler.FactorScheduler(500, 0.9)
    loss = gloss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001, 'lr_scheduler': lrs})
    train(train_iter, net, loss, trainer, batch_size, num_epochs, ctx)


train_fine_tuning(finetune_net, 0.003, batch_size=32, num_epochs=20)


# train_iter = gdata.DataLoader(
#         train_data.transform_first(train_trans), 32, shuffle=True)
# 
# print(type(train_iter._dataset[0][1]))
# print(train_iter._dataset[0][1].dtype)
# # print(train_iter._dataset[0])
# 
# if isinstance(train_iter._dataset[0][0], nd.NDArray):
#     print('TUPLE')


