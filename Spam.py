#Import dataset
import pandas as pd
msgs=pd.read_csv("E:/NLP/SMSSpamCollection",sep='\t',names=['label','msg'])


#data cleaning and preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
L=WordNetLemmatizer()
corpus=[]
for i in range(len(msgs)):
    r=re.sub("[^a-zA-Z^]","",msgs['msg'][i])
    r=r.lower()
    r=r.split()
    
    r=[L.lemmatize(word)for word in r if word not in stopwords.words('english')]
    r=' '.join(r)
    corpus.append(r)
    
#Creating the bagOfWords Model
from sklearn.feature_extraction.text import CountVectorizer
cc=CountVectorizer(max_features=5000)
t=cc.fit_transform(corpus).toarray()


s=pd.get_dummies(msgs['label'])
s=s.iloc[:,1].values

#Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(t,s,test_size=0.20,random_state=0)

#Train the model using naive bayes
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB().fit(X_train,y_train)

y_pred=model.predict(X_test)

#find all correctly detected test values compared to train
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

#Accuracy
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)

