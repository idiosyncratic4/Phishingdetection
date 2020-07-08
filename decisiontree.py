import numpy as np
from sklearn import *
from sklearn import tree
from sklearn.metrics import accuracy_score
training_data=np.genfromtxt('training_dataset.csv',delimiter=',',dtype=np.int32)
inputs=training_data[:,:-1]
outputs=training_data[:,-1]
training_inputs=inputs[:2000]
training_outputs=outputs[:2000]
testing_inputs=inputs[2000:]
testing_outputs=outputs[2000:]
classifier=tree.DecisionTreeClassifier()
classifier.fit(training_inputs,training_outputs)
predictions=classifier.predict(testing_inputs)
accuracy=100*accuracy_score(testing_outputs,predictions)
print("The accuracy of Decision tree on data is"+str(accuracy))
