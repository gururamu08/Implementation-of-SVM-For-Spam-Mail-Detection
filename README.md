# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
-Collect a labeled dataset of emails, distinguishing between spam and non-spam.

-Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

-Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

-Split the dataset into a training set and a test set.

-Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

-Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

-Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

-Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

-Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: GURUMOORTHI R
RegisterNumber:  212222230042
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### 1. Result output

![image](https://github.com/gururamu08/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707009/db9d37c6-2dce-4d6a-a232-8276f23e963a)

### 2. data.head()

![image](https://github.com/gururamu08/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707009/b531f63f-5c20-4cd6-b5d6-ea31dc5e2ef3)


### 3. data.info()

![image](https://github.com/gururamu08/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707009/65778806-dcbf-4d45-943a-e380905b1f28)


### 4. data.isnull().sum()

![image](https://github.com/gururamu08/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707009/667807b8-4c66-4010-9434-c3f76bbbe980)


### 5. Y_prediction value

![image](https://github.com/gururamu08/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707009/acd17207-a670-421d-ae1b-1d615d576d98)


### 6. Accuracy value

![image](https://github.com/gururamu08/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118707009/99ab5791-a3e4-4b9a-b3d1-5125cb443e07)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
