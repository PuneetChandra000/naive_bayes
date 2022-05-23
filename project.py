import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------------

data = pd.read_csv("diabetes.csv")

X = data[["glucose", "bloodpressure"]]
Y = data["diabetes"]

# ------------------------------------ Gaussian NB Algo ----------------------------------------

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.25)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train , Y_train)

predicted_data = classifier.predict(X_test)

print("------------------------------------------------------------------------------------")

print("Accuracy Score using GaussianNB : " , accuracy_score(Y_test , predicted_data))

# ------------------------------- Using Logistic Regression -----------------------------------------------

X = data[["glucose", "bloodpressure"]]
Y = data["diabetes"]

x_train_lr , x_test_lr , y_train_lr , y_test_lr = train_test_split(X , Y , test_size = 0.25 , random_state = 42)

sc_lr = StandardScaler()

x_train_lr = sc_lr.fit_transform(x_train_lr)
x_test_lr = sc_lr.fit_transform(x_test_lr)

classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(x_train_lr , y_train_lr)

predicted_data_lr = classifier_lr.predict(x_test_lr)

print("------------------------------------------------------------------------------------")
print("Accuracy Score using Logistic Regression : " , accuracy_score(y_test_lr , predicted_data_lr))













































































































