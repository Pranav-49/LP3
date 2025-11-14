import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns




url = "emails.csv"
df = pd.read_csv(url)

print("Dataset Loaded Successfully")
print(df.head())
print(df.info())




if 'Email No.' in df.columns:
    df = df.drop('Email No.', axis=1)

X = df.drop('Prediction', axis=1)
y = df['Prediction']

print("\nFeatures and Target Prepared")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n Data Split Successful")



knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)





print("\n Model Evaluation Results ")

print("\nK-Nearest Neighbors:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_knn), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

print("\nSupport Vector Machine:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_svm), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))



acc_knn = accuracy_score(y_test, y_pred_knn)
acc_svm = accuracy_score(y_test, y_pred_svm)

models = ['KNN', 'SVM']
accuracy = [acc_knn, acc_svm]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()