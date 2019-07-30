from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='biking.jpg')
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

plt.show()

