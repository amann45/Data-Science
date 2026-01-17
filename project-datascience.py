# Data science Final project
# LOGISTIC REGRESSION

# 1.Aman Kumar Ray(ACE080BCT010)
# 2.Arjita Yadav(ACE080BCT015)
# 3.Barsha Gyawali(ACE080BCT019)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "Social_Network_Ads.csv"  
df = pd.read_csv(file_path)

# Encode Gender (assuming Male=1, Female=0)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Define independent (X) and dependent (y) variables
X = df[['Gender', 'Age', 'EstimatedSalary']].values  
y = df['Purchased'].values 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add intercept column (bias term)
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

# Mini-batch gradient descent for logistic regression
def mini_batch_GD(X, y, max_iter=1000, batch_size=32, learning_rate=0.01):
    w = np.zeros(X.shape[1])

    for i in range(max_iter):
        batch_indices = np.random.choice(X.shape[0], batch_size, replace=False)
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]

        loss, grad = gradient(batch_X, batch_y, w)
        
        if i % 500 == 0:
            print(f"Loss at iteration {i}: {loss:.4f}")
        
        w -= learning_rate * grad

    return w

def gradient(X, y, w):
    m = X.shape[0]
    h = h_theta(X, w)
    error = h - y
    epsilon = 1e-9  # To prevent log(0) issues
    loss = -np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)) / m
    grad = np.dot(X.T, error) / m
    return loss, grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def h_theta(X, w):
    return sigmoid(np.dot(X, w))

def output(pred):
    return np.round(pred)

# Train the model
w = mini_batch_GD(X_train, y_train, max_iter=5000, batch_size=32, learning_rate=0.01)

# Evaluate the model
yhat = output(h_theta(X_test, w))
print("Accuracy:", accuracy_score(y_test, yhat))

# Function to predict purchase based on user input
def predict_purchase():
    print("\nEnter Details:")
    gender_input = input("Gender (Male/Female): ").strip().lower()
    
    # Convert gender input to numerical value
    if gender_input == "male":
        gender = 1
    elif gender_input == "female":
        gender = 0
    else:
        print("Invalid gender input. Please enter Male or Female.")
        return

    try:
        age = float(input("Age: "))
        ests = float(input("Estimated Salary: "))
    except ValueError:
        print("Invalid input. Please enter numerical values for Age and Estimated Salary.")
        return

    # Preprocess input data
    input_data = np.array([[gender, age, ests]])
    input_scaled = scaler.transform(input_data)  # Apply same normalization
    input_scaled = np.concatenate((np.ones((1, 1)), input_scaled), axis=1)  # Add intercept

    # Predict using the trained model
    prediction = output(h_theta(input_scaled, w))

    # Display the result
    if prediction[0] == 1:
        print("\nPurchased")
    else:
        print("\nNot Purchased")

# Call the function for user input prediction
predict_purchase()
