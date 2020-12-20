import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("bank.csv")

# EDA
fig, ax = plt.subplots()
plt.title('Number of Deposits vs Outcome of previous marketing')
ax.bar([i-.2 for i, _ in enumerate(df[(df["deposit"] == 'yes')]["poutcome"].value_counts().index.tolist())],
        df[(df["deposit"] == 'yes')]["poutcome"].value_counts().tolist(), .4)
ax.bar([i+.2 for i, _ in enumerate(df[(df["deposit"] == 'no')]["poutcome"].value_counts().index.tolist())],
        df[(df["deposit"] == 'no')]["poutcome"].value_counts().tolist(), .4)
plt.xticks([0.0, 1.0, 2.0, 3.0], ['Unknown', 'Success', 'Failure', 'Other'])
plt.legend(['Made deposit', 'Didn\'t make deposit'])
plt.show()

fig, ax = plt.subplots(figsize=(15, 10))
plt.title('Number of depostis for different types of jobs')
ax.bar([i-.2 for i, _ in enumerate(df[(df["deposit"] == 'yes')]["job"].value_counts().index.tolist())],
        df[(df["deposit"] == 'yes')]["job"].value_counts().tolist(), .4)
ax.bar([i+.2 for i, _ in enumerate(df[(df["deposit"] == 'no')]["job"].value_counts().index.tolist())],
        df[(df["deposit"] == 'no')]["job"].value_counts().tolist(), .4)
plt.xticks(list(range(12)), df["job"].value_counts().index.tolist())
plt.legend(['Made deposit', 'Didn\'t make deposit'])
plt.show()

fig, ax = plt.subplots()
plt.title('Number of Deposits vs Level of Education')
ax.bar([i-.2 for i, _ in enumerate(df[(df["deposit"] == 'yes')]["education"].value_counts().index.tolist())],
        df[(df["deposit"] == 'yes')]["education"].value_counts().tolist(), .4)
ax.bar([i+.2 for i, _ in enumerate(df[(df["deposit"] == 'no')]["education"].value_counts().index.tolist())],
        df[(df["deposit"] == 'no')]["education"].value_counts().tolist(), .4)
plt.xticks([0.0, 1.0, 2.0, 3.0], ['Primary', 'Secondary', 'Tertiary', 'Unkown'])
plt.legend(['Made deposit', 'Didn\'t make deposit'])
plt.show()

fig, ax = plt.subplots()
plt.title('Number of Deposits vs Marital Status')
ax.bar([i-.2 for i, _ in enumerate(df[(df["deposit"] == 'yes')]["marital"].value_counts().index.tolist())],
        df[(df["deposit"] == 'yes')]["marital"].value_counts().tolist(), .4)
ax.bar([i+.2 for i, _ in enumerate(df[(df["deposit"] == 'no')]["marital"].value_counts().index.tolist())],
        df[(df["deposit"] == 'no')]["marital"].value_counts().tolist(), .4)
plt.xticks([0.0, 1.0, 2.0], ['Married', 'Single', 'Dicorced'])
plt.legend(['Made deposit', 'Didn\'t make deposit'])
plt.show()

# Preprocessing
for column in "job marital education default housing loan contact month poutcome deposit".split():
    df[column] = CountVectorizer().fit_transform(df[column].tolist()).toarray().tolist()
    df[column] = df[column].apply(lambda x: x.index(1))
    
# Model Building
X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y)

dt_clf = DecisionTreeClassifier().fit(x_train, y_train)
# parameters searching for random forest classifier
params = {'criterion': ('gini', 'entropy'), 
          'max_features': ('auto', 'sqrt', 'log2')}
rf_clf = GridSearchCV(RandomForestClassifier(), 
                      params, scoring='accuracy').fit(x_train, y_train).best_estimator_

nn_model = keras.Sequential([
        layers.Dense(16, activation="relu", name="input"),
        layers.Dense(12, activation="relu", name="hidden"),
        layers.Dense(10, activation="tanh", name="hidden_1"),
        layers.Dense(8, activation="tanh", name="hidden_2"),
        layers.Dense(1, activation="sigmoid", name="output")
])

adam = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
)

nn_model.compile(optimizer=adam, loss=tf.keras.losses.MeanSquaredError())
nn_model.fit(x=x_train, y=y_train, epochs=100)

print("MSE of decision tree: {}".format(MSE(y_test, dt_clf.predict(x_test))))
print("MSE of random forest: {}".format(MSE(y_test, rf_clf.predict(x_test))))
print("MSE of neural network: {}".format(MSE(y_test, nn_model.predict(x_test))))