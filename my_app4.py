from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Uygulamanın başına logo resmi ekleme
st.image("/Users/ebmac/Documents/Clarusway/DS/MLD/Project/car.jpeg", caption="Auto Scout Logo", use_column_width=True)

st.title("Auto Scout Model Prediction with Visual Enhancements")

# Data loading
df = pd.read_csv("Ready_to_ML.csv")

# Sidebar for model selection and training
st.sidebar.title("Model Selection and Training")
model_choice = st.sidebar.selectbox("Choose Model for Prediction", ["Linear Regression", "Random Forest", "Gradient Boosting", "Support Vector Regressor"])

# Makine öğrenimi modelleri sözlüğü
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Regressor": SVR()
}

# Function to train and evaluate multiple models
def train_and_compare_models():
    X = df[['make_model', 'age', 'mileage', 'engine_size']]
    y = df['price']

    # OneHotEncode the 'make_model' column and standardize numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age', 'mileage', 'engine_size']),
            # Set handle_unknown='ignore' to gracefully handle unknown categories
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['make_model'])
        ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a pipeline for each model
    model_performance = {}
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
        model_performance[model_name] = {
            "MSE": mse,
            "R-squared": r2,
            "Cross-validation MSE": np.mean(-cv_scores)
        }
        
        # Save the pipeline instead of just the model
        with open(f"{model_name.replace(' ', '_').lower()}_pipeline.pkl", "wb") as file:
            pickle.dump(pipeline, file)
    
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

if st.sidebar.button("Train and Compare Models"):
    with st.spinner("Training models..."):
        model_performance = train_and_compare_models()
    st.sidebar.success("Model training completed!")
    st.write("### Model Performance Comparison:")
    display_model_performance(model_performance)

# Prediction Input UI
st.write("### Predict Car Price")
make_model = st.selectbox("Select the model of the car:", df['make_model'].unique())
age = st.number_input("Enter the age of the car:", min_value=0, max_value=30, value=5)
mileage = st.number_input("Enter the mileage of the car:", min_value=0, max_value=300000, value=50000)
engine_size = st.number_input("Enter the engine size of the car (in liters):", min_value=0.0, max_value=8.0, value=2.0)

def load_pipeline(model_choice):
    try:
        # Load the selected pipeline
        with open(f"{model_choice.replace(' ', '_').lower()}_pipeline.pkl", "rb") as file:
            pipeline = pickle.load(file)
    except FileNotFoundError:
        # If pipeline file not found, train and save the model
        model_performance = train_and_compare_models()
        pipeline = models[model_choice]
        with open(f"{model_choice.replace(' ', '_').lower()}_pipeline.pkl", "wb") as file:
            pickle.dump(pipeline, file)
    return pipeline

# Predict Price
if st.button("Predict Price"):
    pipeline = load_pipeline(model_choice)
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[make_model, age, mileage, engine_size]], columns=['make_model', 'age', 'mileage', 'engine_size'])
    
    # Make the prediction using the loaded pipeline
    prediction = pipeline.predict(input_data)
    
    st.metric(label="Predicted Price", value=f"€{prediction[0]:.2f}")

    # Plot predictions vs car age
    ages = np.arange(0, 30, 1)
    predictions = [pipeline.predict(pd.DataFrame([[make_model, age, mileage, engine_size]], columns=['make_model', 'age', 'mileage', 'engine_size']))[0] for age in ages]
    
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
