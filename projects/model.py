from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('classification.csv')

# for EDA
# df.corr() 
# df.describe()

c0 = df[df['default'] == 0]
c1 = df[df['default'] == 1]

for feature in df.columns[:-1]:
    plt.figure()
    sns.histplot(c0[feature], color='blue', alpha=0.5, label='Class 0')
    sns.histplot(c1[feature], color='orange', alpha=0.5, label='Class 1')
    plt.show()
    
y = df['default']
x = df[df.columns.difference(['default'])]

xTrain, xTest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 13)
ohe = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
ohe.fit(xTrain['ed'].values.reshape(-1, 1))

xTrain[ohe.get_feature_names_out()] = ohe.transform(xTrain['ed'].values.reshape(-1, 1))
xTest[ohe.get_feature_names_out()] = ohe.transform(xTest['ed'].values.reshape(-1, 1))
xTrain = xTrain.drop(['ed'], axis = 1)
xTest = xTest.drop(['ed'], axis = 1)

pipe = make_pipeline(MinMaxScaler(), LogisticRegression())
pipe.fit(xTrain, ytrain)
pipe.score(xTest, ytest)

y_pred_train = pipe.predict(xTrain)
y_pred_test = pipe.predict(xTest)

# confusion matrix, precision and recall on train and test datasets.    

fp1 = np.array(np.where((ytrain == 0) & (y_pred_train == 1))).size
tp1 = np.array(np.where((ytrain == 1 & (y_pred_train == 1)))).size
fn1 = np.array(np.where((ytrain == 1 & (y_pred_train == 0)))).size
tn1 = np.array(np.where((ytrain == 0 & (y_pred_train == 0)))).size

fp2 = np.array(np.where((ytest == 0) & (y_pred_test == 1))).size
tp2 = np.array(np.where((ytest == 1 & (y_pred_test == 1)))).size
fn2 = np.array(np.where((ytest == 1 & (y_pred_test == 0)))).size
tn2 = np.array(np.where((ytest == 0 & (y_pred_test == 0)))).size

print(f"Accuracy in train: {(tp1 + tn1) / len(ytrain)}, in test: {(tp2 + tn2) / len(ytest)}")
print(f"Precision in train: {tp1 / (tp1 + fp1)}, in test: {tp2 / (tp2 + fp2)}")
print(f"Recall in train: {tp1 / (tp1 + fn1)}, in test: {tp2 / (tp2 + fn2)}")

ax = plt.axes()
df_cm = (confusion_matrix(ytest, y_pred_test, normalize="true")*100).astype(int)

sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = ax )
ax.set_title('Confusion Matrix')
plt.show()

