import os
import numpy as np
import cv2
import shutil

debug = False
numberToHandle = 500

def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_

class ImageData():
    def __init__(self, line, imgDir, image_size=224):
        self.image_size = image_size
        line = line.strip().split()
        assert (len(line) == 207)
        self.list = line
        self.landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, line[196:200])), dtype=np.int32).reshape(-1, 2)
        flag = list(map(int, line[200:206]))
        flag = list(map(bool, flag))
        self.path = os.path.join(imgDir, line[206])
        self.img = None
        self.imgs = []
        self.landmarks = []
        self.boxes = []
        self.ratio = 0.0

    def load_data(self, is_train, repeat):
        img = cv2.imread(self.path)
        height = img.shape[0]  #w*h*c
        width = img.shape[1]
        fy = float(height)/self.image_size
        fx = float(width)/self.image_size
        box = np.array(self.box)
        lmark = np.array(self.landmark)

        if img.shape[0] == 0 or img.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if is_train:
            img = cv2.resize(img, (self.image_size, self.image_size))
            box[:, 1] = self.box[:, 1] / fy
            box[:, 0] = self.box[:, 0] / fx
            lmark[:, 0] = self.landmark[:, 0] / fx
            lmark[:, 1] = self.landmark[:, 1] / fy
            xx = box[1][0] - box[0][0]
            yy = box[1][1] - box[0][1]
            self.ratio = xx * yy / (224 * 224)
        else:
            self.ratio = 1.0

        # print(fx, fy, box, lmark)
        self.imgs.append(img)
        self.landmarks.append(lmark)
        self.boxes = box


        #数据增强
        # if is_train:
        #     while len(self.imgs) < repeat:
        #         angle = np.random.randint(-20, 20)
        #         cx, cy = img.shape[0]/2, img.shape[1]/2  #shape[0]是cols，shape[1]是rows
        #         M, landmark = rotate(angle, (cx,cy), self.landmark)
        #         img = cv2.resize(img, (self.image_size, self.image_size))
        #         self.imgs.append(img)
        #         self.landmarks.append(landmark)



    def save_data(self, path, prefix):
        labels = []
        for i, (img, landmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert landmark.shape == (98, 2)

            # print(self.ratio)
            if self.ratio < 0.05:
                continue

            save_path = os.path.join(path, prefix + '_' + str(i) + '.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)
            box_str = ' '.join(list(map(str, self.boxes.reshape(-1).tolist())))
            landmark_str = ' '.join(list(map(str,landmark.reshape(-1).tolist())))
            # print(self.boxes," :: ", box_str)
            label = '{} {} {}\n'.format(save_path, box_str, landmark_str)
            labels.append(label)
        return labels

def get_dataset_list(imgDir, outDir, landmarkDir, is_train):
    with open(landmarkDir,'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        if debug:
            lines = lines[:numberToHandle]
        for i, line in enumerate(lines):
            Img = ImageData(line, imgDir)
            img_name = Img.path
            Img.load_data(is_train, 1)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i)+'_' + filename)
            labels.append(label_txt)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i+1, len(lines)))

    with open(os.path.join(outDir, 'list.txt'),'w') as f:
        for label in labels:
            f.writelines(label)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    imageDirs = '/home/frank/Desktop/projects/DL/DATA/face/WFLW/WFLW_images'
    Mirror_file = '/home/frank/Desktop/projects/DL/DATA/face/WFLW/Mirror98.txt'

    landmarkDirs = ['/home/frank/Desktop/projects/DL/DATA/face/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
                    '/home/frank/Desktop/projects/DL/DATA/face/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt']

    outDirs = ['test_data', 'train_data']
    for landmarkDir, outDir in zip(landmarkDirs, outDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        if 'list_98pt_rect_attr_test.txt' in landmarkDir:
            is_train = False
        else:
            is_train = True
        imgs = get_dataset_list(imageDirs, outDir, landmarkDir, is_train)

    print('end')