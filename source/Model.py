from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.optimizers import Adam


def getGeneratorModel():
	ngf = 64
	model = Sequential()
	model.add(Dense(input_dim=100, output_dim=(ngf*8)*4*4))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Reshape(dims=(ngf*8, 4, 4)))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(ngf*4, 4, 4, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(ngf*2, 4, 4, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(ngf, 4, 4, border_mode='same'))
	model.add(BatchNormalization())    
	model.add(Activation('relu'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(3, 4, 4, border_mode='same'))
	model.add(Activation('tanh'))
	return model

def getDiscriminatorModel():
	model = Sequential()
	ndf = 64
	model.add(Convolution2D(ndf, 4, 4, subsample=(2, 2), input_shape=(3, 64, 64), border_mode = 'same'))
	model.add(LeakyReLU(0.2))
	# model.add(BatchNormalization())
	model.add(Convolution2D(ndf*2, 4, 4, subsample=(2, 2), border_mode = 'same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
	model.add(Convolution2D(ndf*4, 4, 4, subsample=(2, 2), border_mode = 'same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
	model.add(Convolution2D(ndf*8, 4, 4, subsample=(2, 2), border_mode = 'same'))
	model.add(BatchNormalization())    
	model.add(LeakyReLU(0.2))
	model.add(Convolution2D(1, 4, 4,  border_mode = 'same'))
	model.add(Flatten())
	model.add(Dense(output_dim=1))
	model.add(Activation('sigmoid'))
	return model


def getGeneratorContainingDiscriminator(generator, discriminator):
	model = Sequential()
	model.add(generator)
	discriminator.trainable = False
	model.add(discriminator)
	return model


# return   discriminator_on_generator , generator ,  discriminator 
def getModel():
	generator = getGeneratorModel()
	discriminator = getDiscriminatorModel()
	discriminator_on_generator = getGeneratorContainingDiscriminator(generator , discriminator )

	return discriminator_on_generator , generator ,  discriminator 


def getCompiledModel():
	discriminator_on_generator , generator ,  discriminator  = getModel()
	adam=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
	generator.compile(loss='binary_crossentropy', optimizer=adam)
	discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=adam)
	discriminator.trainable = True
	discriminator.compile(loss='binary_crossentropy', optimizer=adam)

	return discriminator_on_generator , generator ,  discriminator 






