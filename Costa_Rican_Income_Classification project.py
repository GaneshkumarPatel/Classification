#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# 
# 
# # Important features:
# 
#        Target - the target is an ordinal variable indicating groups of income levels.
#         1 = extreme poverty
#         2 = moderate poverty
#         3 = vulnerable households
#         4 = non vulnerable households
#     idhogar - this is a unique identifier for each household. This can be used to create household-wide features, etc. All rows in a given household will have a matching value for this identifier.
#     parentesco1 - indicates if this person is the head of the household.
# 
# 

# In[2]:


df=pd.read_csv('train.csv')


# * As test data provided is not containing target variable and it was used to check for kagale competetion so we ignore test data provided in zip file in our project and we simply train and test our model on train data.

# In[3]:


print(df.shape)


# In[4]:


df.info()


# * we came to know that our data contains 3types of variable. we need to connect object type of variable

# In[5]:


df.select_dtypes('object').head(5)


# * we have only yes or no values insted we can put "0" for "no" and "1" for "yes" in all object variables

# In[6]:


# Reshaping the mixed values for object
mapping_incorrections = {'yes': 1, 'no': 0}

# Reshaping the data based on replace the yes and no
for d in [df]:
    for col in ['dependency', 'edjefa', 'edjefe']:
        d[col] = d[col].replace(mapping_incorrections).astype(np.float64)


# In[7]:


df[['dependency', 'edjefa', 'edjefe']].dtypes


# In[8]:


df.describe()


# In[9]:


df.isnull().any().sum()


# total 5 columns contains missing values there

# In[10]:


df['Target'].isnull().any()


# In[11]:


plt.rcParams['figure.figsize'] = (18, 4)
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')


# In[10]:


def get_dtypes(data, verbose=False):
    """Returns a list: numerical values, object-like columns"""
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics_cols = data.select_dtypes(include=numerics).columns
    object_cols = data.select_dtypes(include='object').columns
    if verbose:
        print('There are {0} numeric cols: {1}\nThere are {2} object cols: {3}\nThere are total cols: {4}'.format
              (len(numerics_cols),numerics_cols, len(object_cols), object_cols, len(data.columns)))
    return [numerics_cols, object_cols]


# In[11]:


def get_missing_values(data, cols):
    """Returns a dataframe with missing values (in absolute and percentage format)"""
    
    missing_percent = data[cols].apply(lambda x: sum(x.isnull())/len(x), axis=0).sort_values(ascending=False)
    missing_abs = data[cols].apply(lambda x: sum(x.isnull()), axis=0).sort_values(ascending=False)
    b = pd.DataFrame({'Missing': missing_abs, 'Percent': missing_percent})
    
    # Removing zero values
    b = b.loc[~(b==0).all(axis=1)]
    return b


# In[12]:


num_cols, obj_cols = get_dtypes(df, verbose=False)


# In[13]:


get_missing_values(df, num_cols)


# # We are having 5 columns where we are getting missing values

# 
# * Notes:
# 
#     Notice that all values that are 0 in the v18q column (binary if a person owns a tablet), for every value that v18q1 is null,  
#     it is also label as 0 means if they dont have a tablet then the tablet count will be zero
# 
# 

# In[14]:


# Filling in the missing value
df['v18q1'] = df['v18q1'].fillna(0)


# * If the house is of own or they are paying loan then there must not be any rent for those people so we are filling all "v2a1" null values to be 0 

# In[15]:


df['v2a1'] = df['v2a1'].fillna(0)


# In[ ]:





# 
# Notes:
# 
#     You cannot be "years" in school if you are not in school.
#     Thus, we can make this value 0 if you are younger than 7 or older than 19
# 

# In[16]:


print(df['rez_esc'].value_counts())


# In[17]:


# If individual is over 19 or younger than 7 and missing years behind, set it to 0
df.loc[((df['age'] > 19) | (df['age'] < 7)) & (df['rez_esc'].isnull()), 'rez_esc'] = 0


# In[20]:


df['Target'].unique()


# In[21]:


df['rez_esc'].isnull().sum()


# we are considering all the missing values in df['rez_esc'] as zero

# In[22]:


df['rez_esc'] = df['rez_esc'].fillna(0)


# In[23]:


df['rez_esc'].isnull().sum()


# In[21]:


sns.boxplot(df['age'])


# In[24]:


df=df.dropna()


# Checking for Distribution and relations of Dataset

# In[25]:


get_missing_values(df, num_cols)


# In[ ]:





# In[28]:


p=sns.pairplot(df.select_dtypes('float64'),diag_kind='kde')


# In[26]:


# Mapping the values of floating columns
from collections import OrderedDict

# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})


# In[30]:


plt.figure(figsize=(20, 8))

# Iterate through the float columns
for i, col in enumerate(df.select_dtypes('float')):
    ax = plt.subplot(3, 3, i + 1)

    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        
        # Plot each poverty level as a separate line
        sns.kdeplot(df.loc[df['Target'] == poverty_level, col].dropna(), 
                    ax=ax, color=color, label=poverty_mapping[poverty_level])
        
    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

plt.subplots_adjust(top=2)


# NOTES: From above plot we came to know that all SQB columns are just a squares of available features and both are highly corelated this will doesn't add any value to our analysis. so we are dropping these unnecessary columns

# In[27]:


# Square features in our dataset
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 
        'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


# In[28]:


# Removing square features as it does not add anything to our dataset
df =df.drop(sqr_, axis=1)
df.shape


# ## Just check poverty level counts

# In[29]:


# SETTING UP THE TARGET VARIABLES
TARGET_AXIS_LABELS = ('non-vunerable hh', 'moderate pov', 'vunerable hh', 'extreme pov')


# In[30]:


# Understanding the distribution of the target variable
countplot=sns.countplot(df.Target)
countplot.set_xticklabels(labels=TARGET_AXIS_LABELS, rotation=40)
plt.title("Poverty Level")


# 
# above graph explains
# 1. Most of the values fall within the non-vunerable.
# 2. Noticed that about 2/3 of the data is considered non-vunerable!
# 

# * Household without head of 

# In[31]:


# Potential error: the household does not have a head leader
hh_head = df.groupby('idhogar')['parentesco1'].sum()

# Find households without a head leader
hh_no_head = df.loc[df['idhogar'].isin(hh_head[hh_head == 0].index), :]
hh_no_head['idhogar'].nunique()


# In[32]:


df.groupby(['idhogar','parentesco1'])['Target'].count().head()


# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier


# In[34]:


X=df.drop(['Id','idhogar','Target'],axis=1)
y=df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)
sc=StandardScaler()
scaledX_train = sc.fit_transform(X_train)
scaledX_test = sc.transform(X_test)


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(scaledX_train,y_train)
pred = knn.predict(scaledX_test)


# In[36]:


### GET both classification report and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(accuracy_score(pred,y_test))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[50]:


rf = RandomForestClassifier(n_estimators=21,random_state=42,max_depth=25)
rf.fit(scaledX_train,y_train)

predicted = rf.predict(scaledX_test)
predtrain = rf.predict(scaledX_train)


# In[51]:


accuracy_score(predtrain,y_train)


# In[52]:


print(accuracy_score(predicted,y_test))
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))


# ## All our F1_score for random forest are more than 80% this is good for our model

# In[42]:


features = list(X_train.columns)
# Feature importances into a dataframe
feature_importances = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
feature_importances.sort_values(by="importance", ascending=False).head(5)


# In[43]:


rftrain=[]
rftest=[]

for j in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=j)
    sc=StandardScaler()
    scaledX_train = sc.fit_transform(X_train)
    scaledX_test = sc.transform(X_test)
    rf = RandomForestClassifier(n_estimators=22)
    rf.fit(scaledX_train,y_train)
          
    rftrain.append(rf.score(scaledX_train,y_train))
    rftest.append(rf.score(scaledX_test,y_test))
print("Compute RF accuracy on the training set")
print(np.mean(rftrain))
print("Compute RF accuracy on the testing set")
print(np.mean(rftest))


# ## Iterative acuuracy for Random forest we get upto 92% now we will verify cross validation score

# In[50]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, scaledX_train,y_train, cv=10)
scores


# In[51]:


print("Accuracy mean score with the 95 percent confidence interval: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Conclusion: 
# * We are getting 92.00% accuracy for iterations with random samples and cross validation Score is 90.00% we could consider this model as good one
# * Random Forest model performs better fo us

# In[54]:


get_ipython().system('pip install shap')


# In[ ]:




