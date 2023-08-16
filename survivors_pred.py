#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rnd


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[4]:


train_df = pd.read_csv("C:\\Users\\Shrreya\\Downloads\\train.csv")
test_df = pd.read_csv("C:\\Users\\Shrreya\\Downloads\\test.csv")
combine= [train_df, test_df]


# In[5]:


print(train_df.columns.values)


# In[6]:


train_df.head()


# In[7]:


train_df.tail()


# In[8]:


train_df.info()


# In[9]:


print('_'*40)


# In[10]:


test_df.info()


# In[11]:


train_df.describe()


# In[12]:


train_df.describe(include=['O'])


# In[13]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[14]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[15]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[16]:


grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep',order=None, hue_order=None)
grid.add_legend()


# In[17]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None,order=None)
grid.add_legend()


# In[18]:


train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]


# In[19]:


print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# In[20]:


for dataset in combine:
 dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex'])


# In[21]:


for dataset in combine:
 dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
 dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
 dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
 dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[22]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}


# In[23]:


for dataset in combine:
 dataset['Title'] = dataset['Title'].map(title_mapping)
 dataset['Title'] = dataset['Title'].fillna(0)


# In[24]:


train_df.head()


# In[25]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# In[26]:


for dataset in combine:
 dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[27]:


train_df.head()


# In[28]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[29]:


guess_ages = np.zeros((2,3))
guess_ages


# In[30]:


for dataset in combine:
 for i in range(0, 2):
    for j in range(0, 3):
        guess_df = dataset[(dataset['Sex'] == i) &                             (dataset['Pclass'] == j+1)]['Age'].dropna()


# In[31]:


age_guess = guess_df.median()


# In[32]:


guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5


# In[33]:


for i in range(0, 2):
    for j in range(0, 3):
        dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]


# In[34]:


dataset['Age'] = dataset['Age'].astype(int)


# In[35]:


train_df.head()

train_df.info()
# In[36]:


train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
train_df.info()


# In[37]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
ascending=True)


# In[38]:


for dataset in combine:
 dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
 dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
 dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
 dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
 dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# In[39]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# In[40]:


for dataset in combine:
 dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'],
as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[41]:


for dataset in combine:
 dataset['IsAlone'] = 0
 dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# In[42]:


train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[43]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
train_df.head()


# In[44]:


for dataset in combine:
 dataset['Age*Class'] = dataset.Age * dataset.Pclass


# In[45]:


train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[46]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[47]:


for dataset in combine:
 dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
ascending=False)


# In[48]:


for dataset in combine:
 dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_df.head()


# In[49]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[50]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
ascending=True)


# In[51]:


for dataset in combine:
 dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
 dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
 dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
 dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
 dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
train_df.head(10)


# In[52]:


test_df.head(10)


# In[53]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[54]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[55]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# In[56]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[57]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[58]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[59]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[60]:


import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
svm_model = SVC(kernel='linear', max_iter=1000)
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[61]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[62]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[63]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[64]:


models = pd.DataFrame({
 'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
 'Random Forest', 'Naive Bayes', 'Perceptron',
 'Stochastic Gradient Decent', 'Linear SVC',
 'Decision Tree'],
 'Score': [acc_svc, acc_knn, acc_log,
 acc_random_forest, acc_gaussian, acc_perceptron,
 acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:





# In[ ]:



 


# In[ ]:




