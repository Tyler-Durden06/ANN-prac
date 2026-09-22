import streamlit as st
import numpy as np
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊",
    layout="centered"
)


# Load trained ANN model with caching for fast performance and low memory usage
@st.cache_resource
def load_ann_model():
    try:
        model = tf.keras.models.load_model("employee_performance_ann.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model 'employee_performance_ann.keras': {e}")
        return None


model = load_ann_model()

# Header Section
st.title("📊 Employee Performance Predictor")
st.write(
    "Predict employee performance based on **Training Hours** and **Attendance Percentage** using an Artificial Neural Network (ANN)."
)
st.markdown("---")

# User inputs in clean layout
col1, col2 = st.columns(2)

with col1:
    training_hours = st.number_input(
        "Training Hours",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=1.0,
        help="Total training hours completed by the employee."
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0,
        help="Attendance percentage of the employee (0 to 100%)."
    )

st.markdown("---")

# Prediction action
if st.button("Predict Performance", type="primary", use_container_width=True):
    if model is None:
        st.error("Model could not be loaded. Please verify 'employee_performance_ann.keras' exists.")
    else:
        with st.spinner("Analyzing performance with Neural Network..."):
            # Prepare input array [training_hours, attendance]
            input_data = np.array([[training_hours, attendance]], dtype=np.float32)

            # Get ANN prediction probability
            raw_pred = model.predict(input_data, verbose=0)
            probability = float(raw_pred[0][0])

            # Classify result based on 0.5 threshold
            result = "Good" if probability >= 0.5 else "Needs Improvement"
            good_pct = round(probability * 100, 2)

        st.subheader("Prediction Results")

        # Visual result indicator
        if result == "Good":
            st.success(f"🎉 **Performance Status: GOOD**")
        else:
            st.warning(f"⚠️ **Performance Status: NEEDS IMPROVEMENT**")

        # Metrics overview
        m1, m2, m3 = st.columns(3)
        m1.metric("Good Probability", f"{good_pct}%")
        m2.metric("Training Hours", f"{training_hours} hrs")
        m3.metric("Attendance", f"{attendance}%")

        # Visual progress bar for score
        st.write("**Performance Score:**")
        st.progress(min(max(probability, 0.0), 1.0))
