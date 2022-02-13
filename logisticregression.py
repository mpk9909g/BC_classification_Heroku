import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
from pickle import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix

# read the train and test files
X_train = pd.read_csv("X_train_scaled.csv")
X_test = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train_categorical.csv")
y_test = pd.read_csv("y_test_categorical.csv")

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

logistic_model = LogisticRegression(random_state = 0)
logistic_model= logistic_model.fit(X_train, y_train)

dump(logistic_model, open('logistic_model.pkl', 'wb'))