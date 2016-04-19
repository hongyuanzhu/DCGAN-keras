
import numpy as np
import sys, glob
import cv2
import os
import os.path
import argparse
import json
import math

import Model
import Vectorize


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type = str , default = "main")
parser.add_argument("--data", type = str)
parser.add_argument("--batch_size", type = int, default = 64 )
parser.add_argument("--epochs", type = int, default = 5 )
args = parser.parse_args()



print "Creating Model"
discriminator_on_generator , generator ,  discriminator = Model.getCompiledModel()
print "Model created"

print "Loading Data"
paths = glob.glob(os.path.join(args.data, "*.jpg"))
print "Data Loaded"




TOTAL_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
EXP_NAME  = args.exp_name

totalBatches =  math.ceil( (len(paths)+0.0) / BATCH_SIZE)

batchesDone = 0 
epochsDone = 0



def saveState():

	global TOTAL_EPOCHS , BATCH_SIZE , EXP_NAME , batchesDone , epochsDone , discriminator_on_generator , generator ,  discriminator

	with open('../data/weights/'+EXP_NAME+'-varstate', 'w') as outfile:
		d = {  "TOTAL_EPOCHS":TOTAL_EPOCHS , "BATCH_SIZE":BATCH_SIZE , "EXP_NAME":EXP_NAME , "batchesDone":batchesDone , "epochsDone":epochsDone , "totalBatches":totalBatches }
		json.dump( d  , outfile)

	generator.save_weights('../data/weights/'+EXP_NAME+'-generator', True)
	discriminator.save_weights('../data/weights/'+EXP_NAME+'-discriminator', True)


def isSavedState():
	global EXP_NAME
	return os.path.isfile('../data/weights/'+EXP_NAME+'-varstate') 


def loadState():
	
	global TOTAL_EPOCHS , BATCH_SIZE , EXP_NAME , batchesDone , epochsDone , discriminator_on_generator , generator ,  discriminator

	with open('../data/weights/'+EXP_NAME+'-varstate') as outfile:
		d = json.load(  outfile)

	TOTAL_EPOCHS = d['TOTAL_EPOCHS']
	BATCH_SIZE = d['BATCH_SIZE']
	EXP_NAME = d['EXP_NAME']
	totalBatches = d['totalBatches']
	batchesDone = d['batchesDone']
	epochsDone = d['epochsDone']

	generator.load_weights('../data/weights/'+EXP_NAME+'-generator')
	discriminator.load_weights('../data/weights/'+EXP_NAME+'-discriminator')




def trainOnBatch(batchNo):
	global TOTAL_EPOCHS , BATCH_SIZE , EXP_NAME , batchesDone , epochsDone , paths , discriminator_on_generator , generator ,  discriminator

	batch = paths[batchNo*BATCH_SIZE : (batchNo+1)*BATCH_SIZE]
	batchSize = len(batch)
	image_batch = Vectorize.getImages(batch)
	noise = Vectorize.getRandomNoise(batchSize)
	generated_images = generator.predict(noise)

	X = np.concatenate((image_batch, generated_images))
	y = [1] * batchSize + [0] * batchSize

	noise = Vectorize.getRandomNoise(batchSize)

	d_loss = discriminator.train_on_batch(X, y)
	g_loss = discriminator_on_generator.train_on_batch(noise, [1]*batchSize )
	print "lodsss" , d_loss , g_loss
	return d_loss , g_loss



if isSavedState():
	loadState()


print "total batches ", totalBatches


while epochsDone < TOTAL_EPOCHS:
	while batchesDone < totalBatches:

		if batchesDone%10 == 0:
			saveState()

		print "Batch ", batchesDone , " Epoch " , epochsDone
		d_loss , g_loss = trainOnBatch(batchesDone)
		print "Generator loss", g_loss, "Discriminator loss", d_loss

		batchesDone += 1
		
	batchesDone = 0 
	epochsDone += 1

print "+++++++++++++++++++++++++"
print "DONE!!! .... wish you best of luck!"


