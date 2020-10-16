# Folk了顺便打颗星星呗【卑微脸】


# Loading data


```python
import pandas as pd
import numpy as np
```


```python
myDf=pd.read_csv("data/test.csv")
```

# Preprocessing


```python
myDf["text"]=myDf["text"].apply(lambda x:x+" <end>")
myDf["tag"]=myDf["tag"].apply(lambda x:x+" END")
myDf[:1]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>将 军 百 战 死 &lt;end&gt;</td>
      <td>B I B I S END</td>
    </tr>
  </tbody>
</table>
</div>




```python
myDf.dropna(inplace=True)
```

# Transforming data to one-hot embedding to generalize X


```python
wordIndexDict={"<pad>":0}
wi=1
for row in myDf["text"].values.tolist():
    if type(row)==float:
        print(row)
        break
    for word in row.split(" "):
        if word not in wordIndexDict:
            wordIndexDict[word]=wi
            wi+=1
vocabSize=wi
```


```python
maxLen=max(len(row) for row in myDf["text"].values.tolist())
sequenceLengths=[len(row) for row in myDf["text"].values.tolist()]
```


```python
myDf["text"]=myDf["text"].apply(lambda x:[wordIndexDict[word] for word in x.split()])
```


```python
import tensorflow as tf
X=tf.keras.preprocessing.sequence.pad_sequences(myDf["text"],
                                                value=wordIndexDict["<pad>"],
                                                padding='post',
                                                maxlen=maxLen)
X
```




    array([[ 1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 1,  2,  7,  8,  9,  4, 10,  6,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 1,  1,  2,  8,  9,  4, 11,  6,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 2,  4,  1,  2,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 1,  2,  4,  1,  2,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0],
           [ 1,  2,  4,  1,  2,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0]])



# Generalizing Y


```python
import tqdm
import re

myDf["tag"]=myDf["tag"].apply(lambda x:re.sub("\-\S+","",x))

tagIndexDict = {"PAD": 0}
ti = 1
for row in tqdm.tqdm(myDf["tag"].values.tolist()):
    for tag in row.split(" "):
        if tag not in tagIndexDict:
            tagIndexDict[tag] = ti
            ti += 1
tagSum = len(list(tagIndexDict.keys()))
myDf["tag"] = myDf["tag"].apply(lambda x:x.split()+["PAD" for i in range(maxLen-len(x.split()))])
myDf["tag"] = myDf["tag"].apply(lambda x:[tagIndexDict[tagItem] for tagItem in x])
# myDf["tag"] = myDf["tag"].apply(lambda x: [[0 if tagI != tagIndexDict[tagItem] else 1
#                                             for tagI in range(len(tagIndexDict))]
#                                             for tagItem in x])
y=np.array(myDf["tag"].values.tolist())
```

    100%|██████████| 6/6 [00:00<?, ?it/s]
    


```python
y.shape # it is OK whether y is one-hot embedding or not
```




    (6, 19)



# Generalizing Model


```python
from BiLSTMCRF import MyBiLSTMCRF
myModel=MyBiLSTMCRF(vocabSize,maxLen, tagIndexDict,tagSum,sequenceLengths)
```


```python
myModel.myBiLSTMCRF.summary()
```

    Model: "sequential_9"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    embedding_9 (Embedding)      (None, 19, 100)           1200
    _________________________________________________________________
    bidirectional_18 (Bidirectio (None, 19, 5)             4240
    _________________________________________________________________
    bidirectional_19 (Bidirectio (None, 19, 5)             440
    _________________________________________________________________
    crf_layer (CRF)              (None, 19)                65
    =================================================================
    Total params: 5,945
    Trainable params: 5,945
    Non-trainable params: 0
    _________________________________________________________________
    

# training model


```python
history=myModel.fit(X,y,epochs=1500)
```

    ......
    Epoch 1496/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.4955
    Epoch 1497/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.4938
    Epoch 1498/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.4922
    Epoch 1499/1500
    6/6 [==============================] - 0s 6ms/sample - loss: 1.4906
    Epoch 1500/1500
    6/6 [==============================] - 0s 7ms/sample - loss: 1.4889
    

# predicting


```python
testI=2
```


```python
preY=myModel.predict(X)[testI]
```


```python
indexTagDict=dict(list(zip(list(tagIndexDict.values()),list(tagIndexDict.keys()))))
indexWordDict=dict(list(zip(list(wordIndexDict.values()),list(wordIndexDict.keys()))))

sentenceList=[indexWordDict[wordItem] for wordItem in X[testI]]
sentenceList=sentenceList[:sentenceList.index("<end>")]

tagList=[indexTagDict[tagItem] for tagItem in preY]
tagList=tagList[:tagList.index("END")]

print(" ".join(sentenceList))
print(" ".join(tagList))
```

    将 将 军 带 上 战 车
    S B I B I B I
    
```
