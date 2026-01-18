import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Page Configuration
st.set_page_config(page_title="KisanMitr: Crop Recommendation", layout="wide", page_icon="🌾")

st.title("🌾 KisanMitr: Smart Crop Recommendation")
st.markdown("""
**AI-Powered Soil Analysis for Indian Farmers.**
Enter your soil health card details to get scientific crop recommendations.
*Data Source: Aligned with National Soil Health Card Schema (AIKosh/Data.gov.in).*
""")

# 2. Load and Train Model
# Cached so it runs fast after the first time
@st.cache_resource
def train_model():
    # Load Dataset from reliable GitHub source
    url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
    df = pd.read_csv(url)
    
    # Split Data
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X_train, y_train)
    
    # Calculate Metrics
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Return model + Test Data for visualization + Full DF for display
    return rf, acc, y_test, y_pred, df

# Train immediately
with st.spinner("Analyzing Soil Data..."):
    model, accuracy, y_test, y_pred, full_df = train_model()

# 3. Sidebar Inputs (Soil Health Card Parameters)
st.sidebar.header("🌱 Soil Health Card Data")

def user_input_features():
    N = st.sidebar.slider('Nitrogen (N)', 0, 140, 90)
    P = st.sidebar.slider('Phosphorus (P)', 5, 145, 42)
    K = st.sidebar.slider('Potassium (K)', 5, 205, 43)
    temperature = st.sidebar.slider('Temperature (°C)', 8.0, 45.0, 20.0)
    humidity = st.sidebar.slider('Humidity (%)', 14.0, 100.0, 82.0)
    ph = st.sidebar.slider('pH Level', 3.5, 10.0, 6.5)
    rainfall = st.sidebar.slider('Rainfall (mm)', 20.0, 300.0, 200.0)
    
    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Main Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Current Soil Profile")
    st.dataframe(input_df, hide_index=True)
    
    if st.button("Predict Best Crop 🚜"):
        # Prediction
        prediction = model.predict(input_df)
        crop = prediction[0]
        
        st.success(f"✅ Recommended Crop: **{crop.upper()}**")
        
        # Contextual Info
        if crop in ['rice', 'jute', 'coffee']:
            st.info("ℹ️ Note: This crop requires high rainfall.")
        elif crop in ['chickpea', 'kidneybeans', 'mothbeans']:
            st.info("ℹ️ Note: These are pulses (Nitrogen fixing crops).")
        elif crop in ['grapes', 'apple']:
            st.info("ℹ️ Note: Suitable for orchard farming.")

with col2:
    st.metric(label="Model Accuracy", value=f"{accuracy:.1%}")
    st.caption("Accuracy calculated on test set.")

# 5. Technical Details (For Interviewers)
st.divider()

# TAB 1: Performance Matrix
with st.expander("📊 View Technical Performance (Confusion Matrix)"):
    st.markdown("### Model Evaluation")
    st.write("This Confusion Matrix shows how the model performed on the Test Set (20% of data). The diagonal line represents correct predictions.")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    st.pyplot(fig)

# TAB 2: Dataset Viewer (Solves the 'Show me the data' problem)
with st.expander("📂 View Raw Dataset (Source Data)"):
    st.markdown("### Crop Recommendation Dataset")
    st.write("This is the raw data used to train the Random Forest model.")
    
    # Show first 100 rows
    st.dataframe(full_df.head(100), height=300)
    
    # Download Button
    csv = full_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset as CSV",
        data=csv,
        file_name='kisan_mitr_dataset.csv',
        mime='text/csv',
    )
