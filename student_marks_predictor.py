import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#step 1
student_data = {
    'Hours_study': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_score': [40, 45, 50, 53, 58, 63, 68, 72, 75, 80]
}

df = pd.DataFrame(student_data)

#step 2
X = df[['Hours_study']]
y = df['Exam_score']

#step 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#step 4
model = LinearRegression()
model.fit(X_train, y_train)

#step 5
user_input = float(input("Enter the number of hours you study: "))
predicted_score = model.predict([[user_input]])

#step 6
print(f"Predicted exam score for {user_input} hours of study: {predicted_score[0]:.2f}")
print(f"Model Accuracy:  { model.score(X_test, y_test)}")
