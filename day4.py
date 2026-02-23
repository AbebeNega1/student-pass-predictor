import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,2,3,4,5,6,7],
    "Attendance":    [40,50,55,60,70,80,85,90,65,75,58,72,88,95],
    "Pass":          [0,0,0,0,1,1,1,1,0,1,0,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied", "Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained!")

prediction = model.predict(X_test)

print("Actual:", y_test.values)
print("Predicted:", prediction)

print("Accuracy:", model.score(X_test, y_test))
new_student = pd.DataFrame({
    "Hours_Studied": [3],
    "Attendance": [85]
})
prediction = model.predict(new_student)

print("New Student Prediction:", prediction)
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")