from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as k
import states


class Cnn(object):
    def __init__(self):
        ##### Set the hyperparameters####

        self.numOfEpochs = 1
        self.batchSize = 128
        self.score = 0
        self.flatten = False
        #################################

        (self.xTrain,self.yTrain),(self.xTest,self.yTest) = mnist.load_data()
        self.preprocessData()

        self.input = InputLayer(input_shape=(self.xTrain.shape[1],self.xTrain.shape[2],self.xTrain.shape[3]))
        #self.input = InputLayer( input_shape=(self.xTrain.shape[1], self.xTrain.shape[2], self.xTrain.shape[3]),\
        #                         input_tensor = (self.xTrain))
        self.model = Sequential()
        self.model.add(self.input)


    def preprocessData(self):
        self.xTrain = self.xTrain.reshape(self.xTrain.shape[0], 28, 28, 1)
        self.xTest = self.xTest.reshape(self.xTest.shape[0], 28, 28, 1)

        self.xTrain = self.xTrain.astype('float32')
        self.xTest = self.xTest.astype('float32')

        self.xTrain /= 255
        self.xTest /= 255

        self.yTrain = np_utils.to_categorical(self.yTrain, 10)
        self.yTest = np_utils.to_categorical(self.yTest, 10)

    def reset(self):
        self.model.summary()
        self.model = Sequential()
        self.model.add(self.input)
        self.score = 0
        self.flatten = False

    def buildModel(self,convNet):
        for eachLayer in convNet:
            if eachLayer.name == states.CONV:
                self.model.add(Convolution2D(nb_filter=eachLayer.numberOfFilters, \
                                             nb_row = eachLayer.filterSize,
                                             nb_col = eachLayer.filterSize,
                                             activation='relu',
                                             padding='SAME'))
                print(states.CONV)

            elif eachLayer.name == states.POOL:
                self.model.add(MaxPooling2D(pool_size=(eachLayer.poolSize,eachLayer.poolSize)))
                                            #strides=eachLayer.strides))
                print(states.POOL)

            elif eachLayer.name == states.FULLY:
                if not self.flatten:
                    self.model.add(Flatten())
                    self.flatten = True
                self.model.add(Dense(eachLayer.numberOfNeurons))
                print(states.FULLY)

            elif eachLayer.name == states.TERMINATE:
                print(states.TERMINATE)
                if not self.flatten:
                    self.model.add(Flatten())
                    self.flatten = True
                #self.model.add(Flatten())
                self.model.add(Dense(10,activation='softmax'))


        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.model.fit(self.xTrain, self.yTrain, batch_size=self.batchSize,
                       nb_epoch=self.numOfEpochs, verbose=1)

        self.score = self.model.evaluate(self.xTest, self.yTest, verbose=0)

        return self.score