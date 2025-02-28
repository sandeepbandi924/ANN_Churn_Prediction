# ANN Churn Prediction

## 📌 Introduction
Customer churn is a significant challenge for businesses, as losing customers leads to revenue loss. This project builds an **Artificial Neural Network (ANN) model** to predict customer churn, helping businesses retain customers by identifying those at risk of leaving.

📌 **Live Demo**: [Streamlit App](https://annchurnprediction.streamlit.app/)

📌 **GitHub Repository**: [ANN Churn Prediction](https://github.com/sandeepbandi924/ANN_Churn_Prediction)

---

## 🚀 Project Workflow

1️⃣ **Data Collection**: Obtained customer data including demographics, transaction history, and service usage.
2️⃣ **Data Preprocessing**: Cleaned missing values, encoded categorical features, and normalized numerical data.
3️⃣ **Model Development**: Built an ANN using TensorFlow/Keras to classify churned and non-churned customers.
4️⃣ **Evaluation**: Measured model performance using accuracy, precision, recall, and F1-score.
5️⃣ **Deployment**: Developed a Streamlit web app for real-time churn prediction.

---

## 📊 Dataset Overview
- **Features**: Customer ID, age, tenure, monthly charges, total charges, contract type, payment method, etc.
- **Target**: Binary classification (Churn = 1, No Churn = 0)

---

## 🔧 Model Training
- **Framework**: TensorFlow/Keras
- **Architecture**:
  - Input layer with feature dimension
  - Hidden layers with ReLU activation
  - Output layer with sigmoid activation
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## 📊 Model Performance
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| ANN | 92% | 90% | 89% | 89.5% |

🔹 **The model achieved 92% accuracy**, making it effective for churn prediction.

---

## 🌍 Deployment
- **Streamlit Web App**: Allows users to input customer details and get churn predictions.
- **Deployment Link**: [ANN Churn Prediction](https://annchurnprediction.streamlit.app/)

```python
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("churn_model.pkl")

def predict_churn(features):
    return model.predict(np.array([features]))

st.title("Customer Churn Prediction")
user_input = st.text_input("Enter customer features")
if st.button("Predict"):
    prediction = predict_churn(user_input)
    st.write("Churn Prediction:", prediction)
```

---

## 🔥 Future Improvements
🔹 Fine-tune ANN hyperparameters for better generalization.<br>
🔹 Incorporate additional customer engagement features.<br>
🔹 Implement a real-time alert system for customer retention strategies.<br>

---

## 📌 Conclusion
This project demonstrates how **ANNs can effectively predict customer churn**, helping businesses retain customers. With a **Streamlit deployment**, users can easily interact with the model in real-time.

📌 **GitHub Repository**: [ANN Churn Prediction](https://github.com/sandeepbandi924/ANN_Churn_Prediction)  
📌 **Live Demo**: [Streamlit App](https://annchurnprediction.streamlit.app/)  

🚀 **Stay ahead of churn!**

