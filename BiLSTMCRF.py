import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from CRF import CRF
# from CRF import CRF

class MyBiLSTMCRF:
    def __init__(self, vocabSize, maxLen, tagIndexDict,tagSum,sequenceLengths=None,vecSize=100,learning_rate=0.01):
        self.vocabSize = vocabSize
        self.vecSize = vecSize
        self.maxLen = maxLen
        self.tagSum = tagSum
        self.sequenceLengths=sequenceLengths
        self.tagIndexDict=tagIndexDict
        self.learning_rate=learning_rate

        self.buildBiLSTMCRF()

    def getTransParam(self,y,tagIndexDict):
        self.trainY=np.argmax(y,axis=-1)
        yList=self.trainY.tolist()
        transParam=np.zeros([len(list(tagIndexDict.keys())),len(list(tagIndexDict.keys()))])
        for rowI in range(len(yList)):
            for colI in range(len(yList[rowI])-1):
                transParam[yList[rowI][colI]][yList[rowI][colI+1]]+=1
        for rowI in range(transParam.shape[0]):
            transParam[rowI]=transParam[rowI]/np.sum(transParam[rowI])
        return transParam
    
    def buildBiLSTMCRF(self):

        myModel=Sequential()
        myModel.add(tf.keras.layers.Input(shape=(self.maxLen,)))
        myModel.add(tf.keras.layers.Embedding(self.vocabSize, self.vecSize))
        myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    self.tagSum, return_sequences=True, activation="tanh"), merge_mode='sum'))
        myModel.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                    self.tagSum, return_sequences=True, activation="softmax"), merge_mode='sum'))
        crf=CRF(self.tagSum,name='crf_layer')
        myModel.add(crf)
        myModel.compile(Adam(learning_rate=self.learning_rate),loss={'crf_layer': crf.get_loss})
        self.myBiLSTMCRF=myModel
        
    def fit(self,X,y,epochs=100,transParam=None):
        if len(y.shape)==3:
            y=np.argmax(y,axis=-1)
        if self.sequenceLengths is None:
            self.sequenceLengths=[row.shape[0] for row in y]
        log_dir = "logs"
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        history=self.myBiLSTMCRF.fit(X,y,epochs=epochs,callbacks=[tensorboard_callback])

        return history

    def predict(self,X):
        preYArr=self.myBiLSTMCRF.predict(X)
        return preYArr

if __name__=="__main__":
    myModel=MyBiLSTMCRF(vocabSize,maxLen, tagIndexDict,tagSum,sequenceLengths)