import cv2
import numpy as np
import os

if __name__ == '__main__':
    # img = cv2.imread("test_data/imgs/0_37_Soccer_soccer_ball_37_45_0.png")
    # img = cv2.imread("test_data/imgs/0_37_Soccer_soccer_ball_37_45_0.png")
    f = open("train_data/list.txt")
    labels = f.readlines()
    imgDir = "train_data/imgs"

    count = 0

    for i, line in enumerate(labels):
        line = line.strip().split()
        # print(len(line))
        img_name = os.path.join(imgDir, line[0])
        img = cv2.imread(img_name)
        # print(img.shape)
        box = np.asarray(list(map(float, line[1:5])), dtype=np.int).reshape(-1, 2)
        landmark = np.asarray(list(map(float, line[5:201])), dtype=np.int).reshape(-1, 2)
        count += 1
        # cv2.rectangle(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (255, 0, 00), 3)
        # for j in range(landmark.shape[0]):
        #     cv2.circle(img, (landmark[j][0], landmark[j][1]), 1, (0, 255, 00), 3)
        #
        # img_id = ''.join(str(i))
        # cv2.namedWindow(img_id)
        # cv2.imshow(img_id, img)
        # cv2.waitKey(5000)

    print(count)