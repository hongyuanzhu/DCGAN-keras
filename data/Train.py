
import numpy as np
import sys, glob
import cv2
import os
import argparse
import json


import Model
import Vectorize


parser = argparse.ArgumentParser()
parser.add_argument("--expname", type = str , default = "main")
parser.add_argument("--data", type = str)
parser.add_argument("--batch_size", type = int, default = 128)
args = parser.parse_args()



print "Creating Model"
discriminator_on_generator , generator ,  discriminator = Model.getCompiledModel()
print "Model created"

print "Loading Data"
paths = glob.glob(os.path.join(path, "*.jpg"))
print "Data Loaded"




TOTAL_EPOCHS = 5
BATCH_SIZE = 64

totalBatches =  int(len(paths) / BATCH_SIZE)

batchesDone = 0 
epochsDone = 0



def saveState():

	with open('data.txt', 'w') as outfile:
		json.dump(data, outfile)

	generator.save_weights('generator', True)
	discriminator.save_weights('discriminator', True)


def loadState():
	pass


def trainOnBatch(batchNo):
	pass


while epochsDone < TOTAL_EPOCHS:
	while batchesDone < totalBatches:
		print "Batch ", batchesDone , " Epoch " , epochsDone

		batchesDone += 1

	epochsDone += 1



