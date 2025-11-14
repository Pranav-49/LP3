import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("diabetes.csv")  
print("Dataset Loaded Successfully")
print(df.head())
print(df.info())


print("\nMissing Values in Each Column:\n", df.isnull().sum())


X = df.drop('Outcome', axis=1)   
y = df['Outcome']              

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nData Split Done")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nEvaluation Metrics ")
print("Confusion Matrix:\n", cm)
print("Accuracy:", round(accuracy, 3))
print("Error Rate:", round(error_rate, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))


plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNN on Diabetes Dataset')
plt.show()