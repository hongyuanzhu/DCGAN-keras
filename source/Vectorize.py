

import numpy as np
import sys, glob
import cv2
import os
import argparse

def load_image(path):
	img = cv2.imread(path, 1)
	img = np.float32(cv2.resize(img, (64, 64))) / 127.5 - 1
	img = np.rollaxis(img, 2, 0)
	return img


def getImages(paths):
	return [ load_image(p) for p in paths  ]