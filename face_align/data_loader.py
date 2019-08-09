import os
import numpy as np


class DataToBatch:

    def __init__(self, path):
        self.batch = []
        self.label = []
        self.len = 0
        self.path = path

    def get_batch(self):
        f = open(self.path)
        lines = f.readlines()
        self.len = len(lines)
        return self.batch