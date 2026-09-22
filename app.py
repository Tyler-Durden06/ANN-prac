import streamlit as st
import numpy as np
import zipfile
import io

# Try importing h5py for lightweight fallback inference
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# Try importing TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊",
    layout="centered"
)


# Load model weights / predictor with caching
@st.cache_resource
def get_predictor():
    """
    Returns a prediction function (training_hours, attendance) -> probability float.
    Uses TensorFlow if available; otherwise uses a lightweight pure NumPy ANN forward pass.
    """
    if TF_AVAILABLE:
        try:
            model = tf.keras.models.load_model("employee_performance_ann.keras")
            def predict_tf(th, att):
                inp = np.array([[th, att]], dtype=np.float32)
                return float(model.predict(inp, verbose=0)[0][0])
            return predict_tf, "TensorFlow Engine"
        except Exception as e:
            st.warning(f"Failed to load via Keras: {e}. Falling back to NumPy engine.")

    # Fallback to NumPy forward pass reading HDF5 weights directly
    if H5PY_AVAILABLE:
        try:
            with zipfile.ZipFile("employee_performance_ann.keras") as z:
                h5_bytes = z.read("model.weights.h5")
            f = h5py.File(io.BytesIO(h5_bytes), 'r')
            w0 = f['layers/dense/vars/0'][:]
            b0 = f['layers/dense/vars/1'][:]
            w1 = f['layers/dense_1/vars/0'][:]
            b1 = f['layers/dense_1/vars/1'][:]
            w2 = f['layers/dense_2/vars/0'][:]
            b2 = f['layers/dense_2/vars/1'][:]
            f.close()

            def predict_numpy(th, att):
                x = np.array([[th, att]], dtype=np.float32)
                h1 = np.maximum(0, np.dot(x, w0) + b0)
                h2 = np.maximum(0, np.dot(h1, w1) + b1)
                y = 1.0 / (1.0 + np.exp(-(np.dot(h2, w2) + b2)))
                return float(y[0][0])

            return predict_numpy, "Lightweight NumPy Engine"
        except Exception as e:
            st.error(f"Failed to load weights: {e}")
            return None, None
    else:
        st.error("Neither TensorFlow nor h5py is installed. Please check requirements.")
        return None, None


predictor, engine_name = get_predictor()

# App Header
st.title("📊 Employee Performance Predictor")
st.write(
    "Enter **Training Hours** and **Attendance Percentage** to predict employee performance using an Artificial Neural Network (ANN)."
)

if engine_name:
    st.caption(f"⚡ *Engine: {engine_name}*")

st.markdown("---")

# User inputs
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
    if predictor is None:
        st.error("Predictor model could not be loaded. Please verify 'employee_performance_ann.keras' exists.")
    else:
        with st.spinner("Analyzing performance with Neural Network..."):
            probability = predictor(training_hours, attendance)
            result = "Good" if probability >= 0.5 else "Needs Improvement"
            good_pct = round(probability * 100, 2)

        st.subheader("Prediction Results")

        if result == "Good":
            st.success("🎉 **Performance Status: GOOD**")
        else:
            st.warning("⚠️ **Performance Status: NEEDS IMPROVEMENT**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Good Probability", f"{good_pct}%")
        m2.metric("Training Hours", f"{training_hours} hrs")
        m3.metric("Attendance", f"{attendance}%")

        st.write("**Performance Score:**")
        st.progress(min(max(probability, 0.0), 1.0))

