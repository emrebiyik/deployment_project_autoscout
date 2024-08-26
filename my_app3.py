import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import pickle
import plotly.graph_objects as go

# Uygulamanın başına logo resmi ekleme
st.image("/Users/ebmac/Documents/Clarusway/DS/MLD/Project/car.jpeg", caption="Auto Scout Logo", use_column_width=True)

st.title("Auto Scout Model Prediction with Visual Enhancements")

# Data loading
df = pd.read_csv("Ready_to_ML.csv")

# Sidebar for model selection and training
st.sidebar.title("Model Selection and Training")
model_choice = st.sidebar.selectbox("Choose Model for Prediction", ["Linear Regression", "Random Forest", "Gradient Boosting", "Support Vector Regressor"])

if st.sidebar.button("Train and Compare Models"):
    with st.spinner("Training models..."):
        model_performance = train_and_compare_models()
    st.sidebar.success("Model training completed!")
    st.write("### Model Performance Comparison:")
    display_model_performance(model_performance)

# Function to train and evaluate multiple models
def train_and_compare_models():
    X = df[['age', 'mileage', 'engine_size']]
    y = df['price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate models
    model_performance = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        
        model_performance[model_name] = {
            "MSE": mse,
            "R-squared": r2,
            "Cross-validation MSE": np.mean(-cv_scores)
        }
    
    return model_performance

# Display model performance comparison with bar plot
def display_model_performance(model_performance):
    performance_df = pd.DataFrame(model_performance).T.reset_index()
    performance_df.columns = ['Model', 'MSE', 'R-squared', 'Cross-validation MSE']
    
    # Bar Plot for MSE
    st.write("### Model MSE Comparison:")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Model', y='MSE', data=performance_df, ax=ax)
    ax.set_title("Model MSE Comparison")
    st.pyplot(fig)

# Prediction Input UI
st.write("### Predict Car Price")
age = st.number_input("Enter the age of the car:", min_value=0, max_value=30, value=5)
mileage = st.number_input("Enter the mileage of the car:", min_value=0, max_value=300000, value=50000)
engine_size = st.number_input("Enter the engine size of the car (in liters):", min_value=0.0, max_value=8.0, value=2.0)

def load_model(model_choice):
    try:
        # Load the selected model
        with open(f"{model_choice.replace(' ', '_').lower()}_model.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        # If model file not found, train and save the model
        model_performance = train_and_compare_models()
        model = models[model_choice]
        with open(f"{model_choice.replace(' ', '_').lower()}_model.pkl", "wb") as file:
            pickle.dump(model, file)
    return model

# Predict Price
if st.button("Predict Price"):
    model = load_model(model_choice)
    input_data = np.array([[age, mileage, engine_size]])
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    st.metric(label="Predicted Price", value=f"€{prediction[0]:.2f}")

    # Plot predictions vs car age
    ages = np.arange(0, 30, 1)
    predictions = [model.predict(np.array([[age, mileage, engine_size]]).reshape(1, -1)) for age in ages]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=predictions, mode='lines+markers', name='Price'))
    fig.update_layout(
        title="Price Predictions vs Car Age",
        xaxis_title="Car Age",
        yaxis_title="Predicted Price (€)",
        plot_bgcolor='#e1e4e8',
        paper_bgcolor='#f0f2f6'
    )
    st.plotly_chart(fig)
