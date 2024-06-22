#!/usr/bin/env python
# coding: utf-8

# # import the libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
from sklearn.svm import SVC
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the dataset

# In[4]:


credit=pd.read_csv("credit card.csv")
print(credit)


# In[5]:


credit.head()


# In[6]:


credit.isnull().sum()


# In[7]:


credit['Class'].value_counts()


# In[8]:


credit.shape


# In[9]:


credit.info()


# In[10]:


credit.describe()


# In[11]:


# separating the data for analysis
legit=credit[credit.Class==0]
fraud=credit[credit.Class==1]


# In[12]:


len(legit)


# In[13]:


len(fraud)


# In[14]:


print(legit.shape)
print(fraud.shape)


# In[15]:


legit.Amount.describe()


# In[16]:


fraud.Amount.describe()


# In[17]:


credit.groupby('Class').mean()


# In[18]:


x=credit['V11'].values
y=credit['V12'].values
x
y


# In[19]:


plt.scatter(x,y,color='blue',label='scatterplot')
plt.title("credict card")
plt.xlabel('V11')
plt.ylabel('V12')
plt.legend(loc=4)
plt.show()
print(x.shape)
print(y.shape)


# In[20]:


x=x.reshape(-1,1)
y=y.reshape(-1,1)
print(x.shape)
print(y.shape)


# # Split the dataset

# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# # Linear Regression

# In[22]:


from sklearn.linear_model import LinearRegression,LogisticRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
a=lm.coef_
b=lm.intercept_
print("Estimated model slope,a:",a)
print("Estimated model slope,b:",b)
lm.predict(x)[0:10]


# In[23]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("Rmse value:{:.4f}".format(rmse))


# In[24]:


from sklearn.metrics import r2_score
y_test_mean=np.full(len(y_test),y_test.mean())
y_test.mean()
#y_test_mean
#r2_score(y_test,y_test_mean)
#r2_score(y_test,y_test)


# # Logistic Regression
# 

# In[25]:


credit.head()


# In[26]:


# last 5 rows of the dataset
credit.tail()


# In[27]:


# dataset information
credit.info()


# In[28]:


# checking the number of missing values in each column
credit.isnull().sum()


# In[29]:


# distribution of legit transactions & fraudulent transactions
credit['Class'].value_counts()


# In[30]:


# separating the data for analysis
legit=credit[credit.Class==0]
fraud=credit[credit.Class==1]


# In[31]:


print(legit.shape)
print(fraud.shape)


# In[32]:


# statistical measures of the data
legit.Amount.describe()


# In[33]:


fraud.Amount.describe()


# In[34]:


# compare the values for both transactions
credit.groupby('Class').mean()


# In[35]:


legit_sample=legit.sample(n=150)


# In[36]:


new_dataset=pd.concat([legit_sample,fraud],axis=0)


# In[37]:


new_dataset.head()


# In[38]:


new_dataset.tail()


# In[39]:


new_dataset['Class'].value_counts()


# In[40]:


new_dataset.groupby('Class').mean()


# In[41]:


x=new_dataset.drop(columns='Class',axis=1)
y=new_dataset['Class']
              
                   


# In[42]:


print(x)


# In[43]:


print(y)


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, y,)


# In[45]:


print(x.shape, x_train.shape, x_test.shape)


# In[46]:


# training the LogisticRegression model with training data
model=LogisticRegression()
model.fit(x_train,y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)


# In[47]:


#accuracy on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[48]:


print('Accuracy on Training data:',training_data_accuracy)


# In[49]:


# accuracy on test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[50]:


print('Accuracy score on Test Data:',test_data_accuracy)


# # Decision Tree

# In[203]:


col_names=['Time','V1','V2','V3','V4','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']
credit.columns=col_names
col_names


# In[204]:


credit.head()


# In[205]:


col_names=['Time','V1','V2','V3','V4','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']
for col in col_names:
    print(credit[col].value_counts())
credit['Class'].value_counts()


# In[206]:


credit.isnull().sum()


# In[207]:


# check data types in X_train
x_train.dtypes


# In[208]:


x_train.head()


# In[209]:


x = credit.iloc[: , 1:30].values
y = credit.iloc[:, 30].values


# In[210]:


print("Input Range : ", x.shape)
print("Output Range : ", y.shape)


# In[211]:


print ("Class Labels : \n", y)


# In[212]:


set_class = pd.value_counts(credit['Amount'], sort = True)

set_class.plot(kind = 'bar', rot=0)

plt.title("Amount Distribution of Transaction")

plt.xlabel("Amount")

plt.ylabel("No of occurences")


# In[213]:


# Correlation
import seaborn as sns
#get correlations of each features in dataset
corrmat = credit.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(credit[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[214]:


stdsc = StandardScaler()
x_train = stdsc.fit_transform(x_train)
x_test = stdsc.transform(x_test)


# In[215]:


print("Training Set after Standardised : \n", x_train[0])


# In[216]:


dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(x_train, y_train)


# In[217]:


y_pred_decision_tree = dt_classifier.predict(x_test)


# In[218]:


com_decision = confusion_matrix(y_test, y_pred_decision_tree)
print("confusion Matrix : \n", com_decision)


# In[219]:


Accuracy_Model = ((com_decision[0][0] + com_decision[1][1]) / com_decision.sum()) *100
print("Accuracy_Decison    : ", Accuracy_Model)

Error_rate_Model= ((com_decision[0][1] + com_decision[1][0]) / com_decision.sum()) *100
print("Error_rate_Decison  : ", Error_rate_Model)

# True Fake Rate
Specificity_Model= (com_decision[1][1] / (com_decision[1][1] + com_decision[0][1])) *100
print("Specificity_Decison : ", Specificity_Model)

# True Genuine Rate
Sensitivity_Model = (com_decision[0][0] / (com_decision[0][0] + com_decision[1][0])) *100
print("Sensitivity_Decison : ", Sensitivity_Model)


# # Support vector machine 

# In[220]:


from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)


# In[221]:


prediction = classifier.predict(x_test) #And finally, we predict our data test.


# In[222]:


class_names=np.array(['0','1']) 


# In[223]:


accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)


# # Random Forest

# In[224]:


print('Valid transaction',len(credit[credit['Amount']==0]))
print('fraud transaction',len(credit[credit['Amount']==1]))


# In[225]:


y= credit['Amount']
x= credit.drop(columns=['Amount'],axis=1)


# In[226]:


# fitting randomforest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()


# In[174]:


#model_1
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
classifier.fit(x_train,y_train)


# In[175]:


y_pred = classifier.predict(x_test)


# In[176]:


from sklearn.metrics import  classification_report, confusion_matrix
print('Classifcation report:\n', classification_report(y_test, y_pred))
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)


# In[177]:


# calculate accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)


# In[ ]:




