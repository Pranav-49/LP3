import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("emails.csv")  
print("Dataset Loaded Successfully")
print(df.head())
print(df.info())


if 'Email No.' in df.columns:
    df = df.drop('Email No.', axis=1)

X = df.drop('Prediction', axis=1)
y = df['Prediction']

print("\nData Prepared Successfully")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nData Split Done")


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)


print("\n📊 Model Evaluation Results 📊")

print("\n🔹 K-Nearest Neighbors Results:")
acc_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy:", round(acc_knn, 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

print("\n🔹 Support Vector Machine Results:")
acc_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy:", round(acc_svm, 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))



models = ['KNN', 'SVM']
accuracy = [acc_knn, acc_svm]

plt.figure(figsize=(6,4))
sns.barplot(x=models, y=accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
