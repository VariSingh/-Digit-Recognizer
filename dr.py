
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('datasets/train.csv').as_matrix()

clf=DecisionTreeClassifier()


#training data
xtrain = data[0:21000,1:]
train_label=data[0:21000,0]

#train model over data
clf.fit(xtrain,train_label)

#test data
xtest = data[21000:,1:]
actual_label=data[21000:,0]



#d=xtest[7]
#d.shape=(28,28)
#pt.imshow(d)
#pt.show()

prediction=clf.predict(xtest)
print(prediction)



