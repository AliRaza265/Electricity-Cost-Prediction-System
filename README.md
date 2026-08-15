# Electricity Cost Prediction System

A Machine Learning regression project that predicts the approximate electricity cost of a site based on its infrastructure and usage-related information.

## 📌 Project Overview

The Electricity Cost Prediction System uses site and resource-consumption information to estimate the monthly electricity cost.

This project was developed as a practical Machine Learning project to apply data preprocessing, feature encoding, feature scaling, regression modeling, evaluation, and deployment using Streamlit.

## 🎯 Project Objective

The main objective of this project is to:

- Analyze electricity-related site information
- Prepare the dataset for Machine Learning
- Convert categorical information into numerical form
- Scale numerical features
- Train a regression model
- Evaluate the model's prediction performance
- Predict electricity costs for new site information
- Provide predictions through a simple user interface

## 📊 Dataset

The project uses a dataset containing information about different sites and their electricity costs.

### Main Features

| Feature | Description |
|---|---|
| `site area` | Area of the site |
| `structure type` | Type of structure |
| `water consumption` | Water consumption of the site |
| `recycling rate` | Recycling rate |
| `utilisation rate` | Utilisation rate |
| `air qality index` | Air quality index |
| `resident count` | Number of residents |
| `electricity cost` | Electricity cost |

The `issue reolution time` feature was removed during preprocessing because it was considered unnecessary for the prediction task.

### Target Variable

`electricity cost`

The target represents the estimated electricity cost.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

## 🔄 Machine Learning Workflow

The project follows this workflow:

1. Load the dataset
2. Understand the dataset
3. Check missing values
4. Check duplicate records
5. Remove an unnecessary feature
6. Encode the categorical `structure type` feature
7. Separate features and target
8. Split the data into training and testing sets
9. Scale the features using MinMaxScaler
10. Train the regression model
11. Evaluate model performance
12. Test the model with sample data
13. Save the trained model and preprocessing objects
14. Use the model for new predictions

## 🤖 Model Used

The final model used in this project is:

**Linear Regression**

A Decision Tree approach was also considered during development, but the final implementation uses Linear Regression.

## 📈 Model Evaluation

The model was evaluated using:

- R² Score
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)

### Final Model Performance

| Metric | Result |
|---|---:|
| R² Score | 0.7717 |
| RMSE | 480.91 |
| MSE | 231,277.33 |

The model achieved an R² score of approximately **77.17%**, meaning it explains a substantial portion of the variation in electricity cost within the test data.

## 🖥️ Application

A Streamlit application is included in the project.

The application allows users to enter site-related information and receive an estimated electricity cost.

### Prediction Output

The system provides an approximate electricity cost based on the information provided by the user.

## 📁 Project Structure

```text
Electricity-Cost-Prediction-System/
│
├── data_set/
│   └── electricity_cost_dataset.csv
│
├── Models/
│   ├── Model.pkl
│   ├── Scaler.pkl
│   └── Encoder.pkl
│
├── main.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
