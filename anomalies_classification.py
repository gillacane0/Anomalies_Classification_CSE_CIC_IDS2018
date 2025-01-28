import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix, zero_one_loss, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import time
from datetime import timedelta, datetime

#READING DATASET
t0 = time.monotonic()
df = pd.read_csv("dataset/cic_custom.csv")
t1 = time.monotonic()
print(f"Loading time: {t1-t0}")
print(df.head())

#LABELS SETTING
dfToReplace = df[['Label_Botnet','Label_Brute-force','Label_DDoS attack','Label_DoS attack','Label_Web attack']]

df['attack'] = dfToReplace.idxmax(axis=1).str.replace('Label_', '')
attacks = df['attack']
mapping = {'Botnet': 0, 'Brute-force':1,'DDoS attack':2, 'DoS attack':3,'Web attack':4}
df['attack'] = df['attack'].replace(mapping)

df = df.drop(['Label_Botnet','Label_Brute-force','Label_DDoS attack','Label_DoS attack','Label_Web attack'], axis=1)

print(df.head())

#PREPROCESSING AND EDA
t = df['attack']
X = df.drop(['attack'],axis=1)

print(df.isnull().sum())
print("DUPLICATED VALUES ",df.duplicated().sum())
#remove duplicared rows values
df = df.drop_duplicates()

feature_names = X.columns.tolist()
class_names = ['Botnet', 'Brute-force','DDoS attack', 'DoS attack','Web attack']

from sklearn.model_selection import train_test_split
X_train,X_test,t_train,t_test = train_test_split(X,t,train_size=0.7,random_state=42)


attacks = attacks.value_counts()
attacks.plot(kind='barh')
plt.xscale('log')

bool_cols = [col for col in X if
               df[col].dropna().value_counts().index.isin([0,1]).all()]
print("Boolean columns:\n")
X[bool_cols].describe().transpose()

numerical_cols = X.drop(bool_cols, axis=1)
print("Numerical columns:\n")
numerical_cols.describe().transpose()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#CORRELATION MATRIX
t0 = time.monotonic()
fig, ax = plt.subplots(figsize=(19, 15))
cax = ax.matshow(abs(X.corr()))

ax.set_xticks(np.arange(X.shape[1]))
ax.set_xticklabels(list(X), fontsize=7, rotation=90)
ax.set_yticks(np.arange(X.shape[1]))
ax.set_yticklabels(list(X), fontsize=7)

cb = fig.colorbar(cax)
cb.ax.tick_params(labelsize=14)

t1 = time.monotonic()
print(f"Correlation matrix building time {t1-t0}")
plt.show()


#importing models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#PRUNING, SEARCHING FOR THE BEST CCP_ALPHA VALUE

t0 = time.monotonic()
ccp_values=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]
clfs = []
for ccp_alpha in ccp_values:
    clf = DecisionTreeClassifier(criterion='entropy',random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train,t_train)
    test_scores = clf.score(X_train,t_train)
    clfs.append(clf)
    print("Number of nodes in the tree with ccp_alpha {} is {} , depth {} or {}, and score {}".format(ccp_alpha, clf.tree_.node_count,  clf.tree_.max_depth, clf.get_depth(), test_scores )  )
t1 = time.monotonic()

print(f"Pruming time: {t1-t0}")


#DECISION TREE
DT = DecisionTreeClassifier(criterion='entropy', random_state=42,max_depth=14, ccp_alpha=0, splitter = 'best')
#Best parameters obtained for Decision tree with geadsearchCV through the commented code below:
"""
#best parameter search combined with pruning (choice of best ccp_alpha)
param_grid_DT = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [10,12,14,16],
    'splitter': ['best','random'],
    #'ccp_alpha': [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05],
}
clf_DT = GridSearchCV(estimator=DT,
                      param_grid=param_grid_DT,
                      scoring='f1_weighted',
                      n_jobs=-1,
                      cv=8,
                      verbose=10,
                      refit=True)

clf_DT.fit(X_train, t_train)
print("Best parameters obtained for Decision tree: ",clf_DT.best_params_)
"""
t0 =  time.monotonic()
DT.fit(X_train,t_train)
t1 = time.monotonic()
print(f"Decision tree training time: {t1-t0}")

print(DT.score(X_train,t_train))
print(DT.score(X_test,t_test))

#PLOTTING THE TREE
t0 = time.monotonic()
fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(DT, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()
t1 = time.monotonic()
print(f"Decision tree plot and show time: {t1-t0}")

#DECISION TREE PREDICTION SCORES
t_predicted_DT = DT.predict(X_test)
results_pruned = confusion_matrix(t_test,t_predicted_DT)
error_pruned = zero_one_loss(t_test,t_predicted_DT)
disp = ConfusionMatrixDisplay(confusion_matrix=results_pruned, display_labels=class_names)
disp.plot()
plt.show()

#RANDOM FOREST
RF = RandomForestClassifier(max_depth = 14,n_estimators=100, criterion='gini', random_state=42)
#Best parameters obtained for Random Forest with geadsearchCV through the commented code below:
"""
parameter_grid_RF = {
    'n_estimators': [100, 101, 102],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 12, 14],
}
clf_RF = GridSearchCV(estimator=RF,
                        param_grid=parameter_grid_RF,
                        scoring='f1_weighted',
                        n_jobs=-1,
                        cv=5,
                        verbose=10,
                        refit=True)
clf_RF.fit(X_train, t_train)
print("Best parameters obtained for Random Forest: ",clf_RF.best_params_)
"""
t0 = time.monotonic()
RF.fit(X_train,t_train)
t1 = time.monotonic()
print(f"Random forest training time: {t1-t0}")

print(RF.score(X_train,t_train))
print(RF.score(X_test,t_test))

#RANDOM FOREST PREDICTION SCORES
t_predicted_RF = RF.predict(X_test)
results_RF = confusion_matrix(t_test, t_predicted_RF)
error_RF = zero_one_loss(t_test, t_predicted_RF)
print(results_RF,'\nError: ',error_RF)
disp_RF = ConfusionMatrixDisplay(confusion_matrix=results_RF, display_labels=class_names)
disp_RF.plot()
plt.show()


#XGB
XGB = xgb.XGBClassifier(objective="multi:softprob", random_state=42, learning_rate=0.1, max_depth = 10)

t0 = time.monotonic()
XGB.fit(X_train, t_train)
t1 = time.monotonic()
print(f"XGBoost training time: {t1-t0}")

print(XGB.score(X_train,t_train))
print(XGB.score(X_test,t_test))

#XGB PREDICTION SCORES
t_predicted_XGB = XGB.predict(X_test)
results_XGB = confusion_matrix(t_test, t_predicted_XGB)
error_XGB = zero_one_loss(t_test, t_predicted_XGB)
disp_XGB = ConfusionMatrixDisplay(confusion_matrix=results_XGB, display_labels=class_names)
disp_XGB.plot()
plt.show()



