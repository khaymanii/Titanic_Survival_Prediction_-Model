
# Importing the Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Data collection and processing

titanic_data = pd.read_csv('titanic.csv')

titanic_data.head(10)

titanic_data.shape

titanic_data.info()

titanic_data.isnull().sum()

# Handling the missing values : dropping the Cabin column

titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# Replacing the missing values in age column with the mean value

titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# Replacing the missing values in embarked column with the mode value

print(titanic_data['Embarked'].mode())

print(titanic_data['Embarked'].mode()[0])

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

titanic_data.isnull().sum()

# Data analysis

titanic_data.describe()

# Finding the number of people who survived and not survived

titanic_data['Survived'].value_counts()

# Data Visualization

sns.set()

# Making a countplot for 'Survived' column

sns.countplot('Survived', data = titanic_data)

# Making a countplot for 'Sex' column

sns.countplot('Sex', data = titanic_data)

# Number of survivors gender wise

sns.countplot('Sex', hue='Survived', data=titanic_data)

# Making a countplot for 'Pclass' column

sns.countplot('Pclass', data = titanic_data)

# Number of survivors pclass wise

sns.countplot('Pclass', hue='Survived', data=titanic_data)

# Encoding the categorical column

titanic_data['Sex'].value_counts()

titanic_data['Embarked'].value_counts()

# Converting th categorical columns

titanic_data.replace({'Sex': {'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)

titanic_data.head()

# Splitting Features and Target data

X = titanic_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Survived'], axis = 1)
y = titanic_data['Survived']

print(X)
print(y)


# Splitting into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)



model = LogisticRegression()

model.fit(X_train, y_train)


# Model Evaluation : Training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# Model Evaluation : Test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accuracy score of training data : ', test_data_accuracy)


