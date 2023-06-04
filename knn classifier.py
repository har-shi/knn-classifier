import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings('ignore')

data = r'E:\project list practice\KNN classifier\breast-cancer-wisconsin.txt'
df = pd.read_csv(data, header= None)
df.shape
df.head()
# Rename Column Name
col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

df.columns = col_names

df.columns
df.head()

# drop id column from dataset
df.drop('Id', axis=1, inplace= True)
df.info()

for var in df.columns:
    print(df[var].value_counts())

# Convert data type of Bare_Nuclei to integer
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
df.dtypes

# Missing value in variables
df.isnull().sum()
# check Na value in dataset
df.isna().sum()

# check frequency distribution
df['Bare_Nuclei'].value_counts()
# check unique values
df['Bare_Nuclei'].unique()
# check for nan value
df['Bare_Nuclei'].isna().sum()
# check frequency distribution of values in 'class' variable
df['Class'].value_counts()
# view percentage of frequency distribution of values in `Class` variable

df['Class'].value_counts()/np.float(len(df))

# view summary statistics in numerical variables
print(round(df.describe(),2))

# data visualization
# plot histogram of the variable
plt.rcParams['figure.figsize']=(30,25)
df.plot(kind='hist' , bins=10, subplots=True, layout=(5,2), sharex=False, sharey=False)
plt.show()

correlation = df.corr()
correlation['Class'].sort_values(ascending=False)

# interpretation.
# correlation heat map
plt.figure(figsize=(10,8))
plt.title('Correlation of attributes with class variable')
a = sns.heatmap(correlation, square=True, annot=True, fmt=' .2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.show()

# Declare feature vector and target variable
 X = df.drop(['Class'], axis=1)
y = df['Class']

# spliting X and y into training and testing phase
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 0)

# check the shape of X_train X_test
X_train.shape, X_test.shape 

# Feature engineering, datatype X_train
X_train.dtypes

# check missing values in numerical variables in X_train
X_train.isnull().sum()    

# check missing values in numerical variables in X_train
X_test.isnull().sum()    

# print percentage of missing values in training set
 for col in X_train.columns:
     if X_train[col].isnull().mean()>0:
         print(col, round(X_train[col].isnull().mean(),4))
         
# impute missing values in X_train and X_test with respective column median in X_train
for df1 in [X_train, X_test]:
    for col in X_train.columns:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)
# check missing values in numerical variable in X_train
X_train.isnull().sum()
# check missing value in X_test
X_test.isnull().sum()

X_train.head()
X_test.head()
# Feature Scaling
cols = X_train.columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train= pd.DataFrame(X_train, columns=[cols])
X_test= pd.DataFrame(X_test, columns=[cols])
X_train.head()

# fit KNN classifier
# import knn classifier
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=3)


# fit the model to the training set
knn.fit(X_train, y_train)

# predict test set result
y_pred =knn.predict(X_test)
y_pred

# probability of getting output as 2 - benign cancer

knn.predict_proba(X_test)[:,0]
# probability of getting output as 4 - malignant cancer

knn.predict_proba(X_test)[:,1]

# check accuracy result
from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# compair train-set and test-set accuracy
y_pred_train = knn.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'.format (accuracy_score(y_train, y_pred_train,)) )

# check for overfiting and underfiting
# print the score on training test set
print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))
# check class distribution in test set
y_test.value_counts()

# check null accuracy score
null_accuracy = (85/(85+55))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# instantiate the model with k=5
knn_5 = KNeighborsClassifier(n_neighbors=5)
# fit the model to the training set 
knn_5.fit(X_train, y_train)
# predict on the test-set
y_pred_5 = knn_5.predict(X_test)
print('model accuracy score with k=5 : {0:0.4f}'.format(accuracy_score(y_test, y_pred,))  )

# rebuild knn classification model using k = 6
# instantiate the model with k=6
knn_6 = KNeighborsClassifier(n_neighbors=6)
# fit the model to the training set 
knn_6.fit(X_train, y_train)
# predict on the test-set
y_pred_6 = knn_6.predict(X_test)
print('model accuracy score with k=6 : {0:0.4f}'.format(accuracy_score(y_test, y_pred_6,))  )
# instantiate the model with k=7
knn_7 = KNeighborsClassifier(n_neighbors=7)
# fit the model to the training set 
knn_7.fit(X_train, y_train)
# predict on the test-set
y_pred_7 = knn_5.predict(X_test)
print('model accuracy score with k=7 : {0:0.4f}'.format(accuracy_score(y_test, y_pred_7,))  )
# instantiate the model with k=8
knn_8 = KNeighborsClassifier(n_neighbors=8)
# fit the model to the training set 
knn_8.fit(X_train, y_train)
# predict on the test-set
y_pred_8 = knn_8.predict(X_test)
print('model accuracy score with k=8 : {0:0.4f}'.format(accuracy_score(y_test, y_pred_8,))  )
# instantiate the model with k=9
knn_9 = KNeighborsClassifier(n_neighbors=9)
# fit the model to the training set 
knn_9.fit(X_train, y_train)
# predict on the test-set
y_pred_9 = knn_9.predict(X_test)
print('model accuracy score with k=9 : {0:0.4f}'.format(accuracy_score(y_test, y_pred_9,))  )

# confusion matrix
# Print the Confusion Matrix with k =3 and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# Print the Confusion Matrix with k =7 and slice it into four pieces

cm_7 = confusion_matrix(y_test, y_pred_7)

print('Confusion matrix\n\n', cm_7)

print('\nTrue Positives(TP) = ', cm_7[0,0])

print('\nTrue Negatives(TN) = ', cm_7[1,1])

print('\nFalse Positives(FP) = ', cm_7[0,1])

print('\nFalse Negatives(FN) = ', cm_7[1,0])

# visualize confusion matrix with seaborn heatmap

plt.figure(figsize=(6,4))

cm_matrix = pd.DataFrame(data=cm_7, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# classification matrix
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_7))

# classification accuracy
TP = cm_7[0,0]
TN = cm_7[1,1]
FP = cm_7[0,1]
FN = cm_7[1,0]
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# classification error
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
# true positive rate
true_positive_rate = TP / float(TP + FN)

print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
# false positive rate
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
# specificity
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))

# print the first 10 predicted probabilities of two classes- 2 and 4
y_pred_prob = knn.predict_proba(X_test)[0:10]
y_pred_prob
# store the probabilities in dataframe
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - benign cancer (2)', 'Prob of - malignant cancer (4)'])

y_pred_prob_df

# print the first 10 predicted probabilities for class 4 - Probability of malignant cancer
knn.predict_proba(X_test)[0:10, 1]
# store the predicted probabilities for class 4 - Probability of malignant cancer
y_pred_1 = knn.predict_proba(X_test)[: , 1]

# plot histogram of predicted probabilities


# adjust figure size
plt.figure(figsize=(6,4))


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred_1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of malignant cancer')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of malignant cancer')
plt.ylabel('Frequency')

# ROC AUC
# plot ROC curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_1, pos_label=4)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Breast Cancer kNN classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred_1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(knn_7, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

# Applying 10-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn_7, X_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))












































