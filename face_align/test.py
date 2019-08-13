import os
import numpy as np
import random
import d2lzh as d2l
import cv2
import mxnet as mx
from mxnet import image
from mxnet.gluon import data as gdata


gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.


def f():
    a = mx.nd.ones((100,100))
    b = mx.nd.ones((100,100))
    e = mx.nd.array(a)
    c = a + b
    print(e)
# in default mx.cpu() is used
f()
# change the default context to the first GPU
with mx.Context(gpu_device):
    f()