# =====================================================
# CUSTOMER CHURN PREDICTION WEB APP
# USING MULTIPLE MACHINE LEARNING ALGORITHMS
# + STREAMLIT USER INTERFACE
# =====================================================

# =====================================================
# IMPORT LIBRARIES
# =====================================================

import streamlit as st

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# =====================================================
# PAGE TITLE
# =====================================================

st.title("Customer Churn Prediction System")

st.write(
    "This application predicts whether "
    "a customer will churn or stay."
)

# =====================================================
# LOAD DATASET
# =====================================================

# Install dependency:
# pip install kagglehub[pandas-datasets]

import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "Dataset.csv"

df=pd.read_csv(file_path)

# =====================================================
# SHOW FIRST 5 ROWS
# =====================================================

st.subheader("First 5 Rows of Dataset")

st.dataframe(df.head())

# =====================================================
# BASIC INFORMATION
# =====================================================

st.subheader("Dataset Information")

st.write(df.head())

st.subheader("Missing Values")

st.write(df.isnull().sum())

# =====================================================
# DATA PREPROCESSING
# =====================================================

# Remove customerID column

df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges into numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors='coerce'
)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Create CustomerCategory feature

df['CustomerCategory'] = np.where(
    df['tenure'] < 12,
    'New Customer',
    'Old Customer'
)

st.subheader("New Feature Created")

st.write(
    df[['tenure', 'CustomerCategory']].head()
)


# =====================================================
# EDA VISUALIZATIONS
# =====================================================

# -----------------------------------------------------
# 1. CHURN COUNT GRAPH
# -----------------------------------------------------

st.subheader("Customer Churn Count")

fig1, ax1 = plt.subplots(figsize=(6,5))

sns.countplot(
    x='Churn',
    data=df,
    ax=ax1
)

ax1.set_title("Customer Churn Count")

st.pyplot(fig1)

# -----------------------------------------------------
# 2. CONTRACT TYPE VS CHURN
# -----------------------------------------------------

st.subheader("Contract Type vs Churn")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.countplot(
    x='Contract',
    hue='Churn',
    data=df,
    ax=ax2
)

ax2.set_title("Contract Type vs Churn")

plt.xticks(rotation=10)

st.pyplot(fig2)

# -----------------------------------------------------
# 3. MONTHLY CHARGES VS CHURN
# -----------------------------------------------------

st.subheader("Monthly Charges vs Churn")

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.boxplot(
    x='Churn',
    y='MonthlyCharges',
    data=df,
    ax=ax3
)

ax3.set_title("Monthly Charges vs Churn")

st.pyplot(fig3)


# =====================================================
# LABEL ENCODING
# =====================================================

# Store encoders for later use
label_encoders = {}

for column in df.columns:

    if df[column].dtype == 'object':

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column].astype(str))

        label_encoders[column] = le

# =====================================================
# FORCE CONVERT ALL COLUMNS
# =====================================================

for column in df.columns:

        df[column] = pd.to_numeric(df[column],errors='coerce')

print(df.dtypes)


# Fill missing values

df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

# =====================================================
# DEFINE FEATURES AND TARGET
# =====================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# CREATE MACHINE LEARNING MODELS
# =====================================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(),

    "KNN":
        KNeighborsClassifier()
}

# =====================================================
# TRAIN AND EVALUATE MODELS
# =====================================================

results = {}

st.subheader("Model Performance")

for name, model in models.items():

    # Train model

    model.fit(X_train, y_train)

    # Predictions

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    # Accuracy

    train_accuracy = accuracy_score(
        y_train,
        y_train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    # Store accuracy

    results[name] = test_accuracy

    # Display Results

    st.write("===================================")

    st.write("MODEL:", name)

    st.write(
        "Train Accuracy:",
        round(train_accuracy * 100, 2),
        "%"
    )

    st.write(
        "Test Accuracy:",
        round(test_accuracy * 100, 2),
        "%"
    )

    # Overfitting / Underfitting

    if train_accuracy > test_accuracy + 0.10:

        st.warning(
            "Model may be OVERFITTING"
        )

    elif train_accuracy < 0.70 and test_accuracy < 0.70:

        st.warning(
            "Model may be UNDERFITTING"
        )

    else:

        st.success(
            "Model is performing well"
        )

# =====================================================
# FINAL MODEL COMPARISON
# =====================================================

results_df = pd.DataFrame({

    'Model': results.keys(),

    'Accuracy': results.values()
})

st.subheader("Final Accuracy Comparison")

st.dataframe(results_df)

# =====================================================
# FINAL ACCURACY GRAPH
# =====================================================

fig4, ax4 = plt.subplots(figsize=(10,6))

sns.barplot(
    x='Model',
    y='Accuracy',
    data=results_df,
    ax=ax4
)

for container in ax4.containers:
  ax4.bar_label(container)

ax4.set_title("Model Accuracy Comparison")

plt.xticks(rotation=15)

plt.ylim(0,1)

st.pyplot(fig4)

# =====================================================
# BEST MODEL
# =====================================================

best_model = results_df.loc[
    results_df['Accuracy'].idxmax()
]

st.subheader("Best Performing Model")

st.write(
    "Best Model:",
    best_model['Model']
)

st.write(
    "Accuracy:",
    round(best_model['Accuracy'] * 100, 2),
    "%"
)

# =====================================================
# SELECT LOGISTIC REGRESSION MODEL
# =====================================================

trained_model = models["Logistic Regression"]

# =====================================================
# USER INPUT SECTION
# =====================================================

st.header("Enter Customer Details")

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

Dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0
)

PhoneService = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Bank transfer",
        "Credit card",
        "Electronic check",
        "Mailed check"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0
)

# =====================================================
# MANUAL ENCODING
# =====================================================

gender = 0 if gender == "Female" else 1

Partner = 0 if Partner == "No" else 1

Dependents = 0 if Dependents == "No" else 1

PhoneService = 0 if PhoneService == "No" else 1

MultipleLines_mapping = {
    "No": 0,
    "Yes": 1,
    "No phone service": 2
}

MultipleLines = MultipleLines_mapping[
    MultipleLines
]

InternetService_mapping = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

InternetService = InternetService_mapping[
    InternetService
]

OnlineSecurity = 0 if OnlineSecurity == "No" else 1

OnlineBackup = 0 if OnlineBackup == "No" else 1

DeviceProtection = 0 if DeviceProtection == "No" else 1

TechSupport = 0 if TechSupport == "No" else 1

StreamingTV = 0 if StreamingTV == "No" else 1

StreamingMovies = 0 if StreamingMovies == "No" else 1

Contract_mapping = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

Contract = Contract_mapping[Contract]

PaperlessBilling = (
    0 if PaperlessBilling == "No"
    else 1
)

PaymentMethod_mapping = {
    "Bank transfer": 0,
    "Credit card": 1,
    "Electronic check": 2,
    "Mailed check": 3
}

PaymentMethod = PaymentMethod_mapping[
    PaymentMethod
]

# =====================================================
# CUSTOMER CATEGORY
# =====================================================

if tenure < 12:

    CustomerCategory = 0

else:

    CustomerCategory = 1

# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button("Predict Churn"):

    user_data = [[

        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges,
        CustomerCategory
    ]]

    # Predict

    prediction = trained_model.predict(
        user_data
    )

    # =================================================
    # DISPLAY RESULT
    # =================================================

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error(
            "Customer is likely to CHURN"
        )

    else:

        st.success(
            "Customer is likely to STAY"
        )