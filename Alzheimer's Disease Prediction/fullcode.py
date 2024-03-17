# %%
from IPython.display import clear_output


clear_output()

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split,cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, confusion_matrix, ConfusionMatrixDisplay ,recall_score, precision_score, f1_score, classification_report

import xgboost as xgb
from lightgbm import LGBMClassifier
import lazypredict
from lazypredict.Supervised import LazyClassifier

import time
import warnings
warnings.filterwarnings('ignore')

# %%
from IPython.display import clear_output


clear_output()

# %%
from IPython.display import clear_output


clear_output()

# %%
from IPython.display import clear_output


clear_output()

# %%
df = pd.read_csv("https://github.com/adityaram-2003/Datasets/blob/main/Alzheimer's/data.csv")

# %%
df.sample(5)

# %%
null_= pd.DataFrame(df.isna().sum())
sum(null_)

# %%
df.select_dtypes("O").values[:5]

# %%
df.drop(columns=["ID"],inplace=True)

# %%
# plt.figure()
# sns.countplot(x = "class", data=df)

# %%
df["class"] = [1 if i == "P" else 0 for i in df["class"]]

# %%
correlation = df.corr().abs()["class"].drop("class")
print(correlation.sort_values(ascending=False))

# %%
variances = df.var()
threshold = 0.2
low_variance = variances[variances <= threshold].index
filtered_data = df.drop(columns = low_variance)
print("Column count before variance threshold: ",df.shape[1])
print("Column count after  variance threshold: ",filtered_data.shape[1])

# %%
X = df.drop(columns="class")
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, shuffle=True, random_state=3)

# Scale X

"""
That point extremly important to prevent information leakedge!

"""

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

scaler1.fit(X_train)
scaler2.fit(X_test)

X_train = pd.DataFrame(scaler1.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test  = pd.DataFrame(scaler2.transform(X_test),  index=X_test.index,  columns=X_test.columns)


print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)

# %%
# Dimensionality reduction using PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# %%
clf = LazyClassifier(verbose=0,
                     ignore_warnings=True,
                     custom_metric=None,
                     predictions=False,
                     random_state=3,
                     classifiers='all')

models, predictions = clf.fit(X_train , X_test , y_train , y_test)
clear_output()

# %%
models

# %%
line = px.line(data_frame= models ,y =["Accuracy"] , markers = True)
line.update_xaxes(title="Model",
              rangeslider_visible = False)
line.update_yaxes(title = "Accuracy")
line.update_traces(line_color="red")
line.update_layout(showlegend = True,
    title = {
        'text': 'Accuracy vs Model',
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

line.show()

# %%
line = px.line(data_frame= models ,y =["Time Taken"] , markers = True)
line.update_xaxes(title="Model",
              rangeslider_visible = False)
line.update_yaxes(title = "Time(s)")
line.update_traces(line_color="purple")
line.update_layout(showlegend = True,
    title = {
        'text': 'TIME TAKEN vs Model',
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

line.show()

# %%
from sklearn.ensemble import ExtraTreesClassifier

et_classifier = ExtraTreesClassifier(n_estimators=100, max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=True)

et_classifier.fit(X_train, y_train)

y_pred = et_classifier.predict(X_test)

scores = cross_val_score(et_classifier, X_train, y_train, cv=5)  # cv=5 for 5-fold cross-validation

# print("Cross-Validation Scores:", scores)
# print("Mean CV Score:", scores.mean())
# print("Standard Deviation of CV Scores:", scores.std())

# Define the randomized algorithm function
def randomized_algorithm():
    # Implementing a simple randomized algorithm
    # Generating a random number between 0 and 1
    random_number = np.random.rand()
    # Setting a threshold for decision
    threshold = 0.5
    # Checking if the random number is greater than the threshold
    if random_number > threshold:
        return 'Yes, the person has Alzheimers'
    else:
        return 'No the person does not have Alzheimers'

# Create a plot
plt.plot()

# Generate randomized 'Yes' or 'No'
random_result = randomized_algorithm()

# Add text annotation with the randomized result
plt.text(0, 0, random_result, fontsize=12, ha='center')

# Remove numbers on x-axis and y-axis
plt.xticks([])
plt.yticks([])

# Show the plot
plt.show()
# # %%
# from sklearn import metrics
# #confusion_matrix =confusion_matrix(y_test, y_pred)
# #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
# plt.figure(figsize = (10,5))
# #cm_display.plot()
# plt.show()

# %%
# Checking the F1 score of the model
f1 = f1_score(y_test, y_pred)
print(f1)

# %%
# Calculating precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)


