import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC


trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

submission = pd.DataFrame()
submission["PassengerId"] = testData["PassengerId"]

trainData["Sex"].replace('male', '0', True)
trainData["Sex"].replace('female', '1', True)
testData["Sex"].replace('male', '0', True)
testData["Sex"].replace('female', '1', True)


columns = ["Pclass","Sex","Age","SibSp","Parch","Fare"]
dataToTrain = trainData[columns].fillna(-1000)
y = trainData["Survived"]
X = dataToTrain
XTrain, XTest, yTrain, yTest = tts(X, y, test_size=0.3, random_state=21, stratify=y)

model = SVC()
model.fit(XTrain, yTrain)

print(model.score(XTest,yTest))

dataToTest = testData[columns].fillna(-1000)
submission['Survived'] = model.predict(dataToTest)
submission.to_csv("Submission.csv",index=False)
