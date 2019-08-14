from mxnet.gluon import data as gdata
from mxnet import image
import os
import mxnet.ndarray as nd
import numpy as np
import warnings


class RegressionDataSet(gdata.dataset.Dataset):
    """A dataset for loading image files and regression points specified by a given list.

        like::
        in list.txt, the first element is image filename, the rest are boundingbox and landmark coordinates.
            /home/path/to/img 10 20 30 40 ...

        Parameters
        ----------
        list_file : str
            Path to list file directory.
        img_folder:
            Path to the img flie directory if there is only filename in list_file.
            if   in list_file, the full filename is provided, then set img_folder to None by default
            else only the filename itself is provided in list_file, then set img_folder as the directory contains the imgs
        flag : {0, 1}, default 1
            If 0, always convert loaded images to greyscale (1 channel).
            If 1, always convert loaded images to colored (3 channels).
        transform : callable, default None
            A function that takes data and label and transforms them::

                transform = lambda data, label: (data.astype(np.float32)/255, label)

        Attributes
        ----------
        items : list of tuples
            List of all images in (filename, label) pairs.
        """

    def __init__(self, list_file, img_folder = None, flag=1, transform=None):
        self._root = os.path.expanduser(list_file)
        self._flag = flag
        self._imgfolder = img_folder
        self._transform = transform
        self._list_images(self._root)

    def _list_images(self, root):
        self.items = []

        f = open(self._root)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            filename = line[0]
            #the conversion cannot be ignored, we can not store list as label
            lab = np.array(line[1:])
            label = nd.array(lab)
            if os.path.isfile(filename):
                self.items.append((filename, label))
            else:
                filename = os.path.join(self._imgfolder, filename)
                self.items.append((filename, label))
                # if os.path.exists(filename):
                #     warnings.warn('Ignoring %s, which does not exsit.' %filename, stacklevel=3)
                #     continue
                # else:
                #     self.items.append((filename, label))

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)