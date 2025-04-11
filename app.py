import streamlit as st
import numpy as np
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df = np.array(df)
    np.random.shuffle(df)
    m = df.shape[0]
    train_data = df[:int(0.8 * m), :]
    val_data = df[int(0.8 * m):, :]

    X_val = val_data[:, 1:].T / 255.0
    y_val = val_data[:, 0]
    return X_val, y_val

# loading weights from model
@st.cache_data
def load_weights():
    weights = np.load("trained_weights.npz")
    return weights["W1"], weights["B1"], weights["W2"], weights["B2"]

def ReLU(X):
    return np.maximum(X, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1 @ X + B1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + B2
    A2 = softmax(Z2)
    return A2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Load data and weights
X_val, y_val = load_data()
W1, B1, W2, B2 = load_weights()

# streamlit interface
st.title("MNIST Digit Classifier")

if "index" not in st.session_state:
    st.session_state.index = 0

col1, col2 = st.columns([4, 1])
with col1:
    st.session_state.index = st.slider("Select an image index:", 0, X_val.shape[1] - 1, st.session_state.index)
with col2:
    if st.button("🎲 Random"):
        st.session_state.index = np.random.randint(0, X_val.shape[1])

index = st.session_state.index
x = X_val[:, index].reshape(784, 1)
actual = y_val[index]
A2 = forward_propagation(W1, B1, W2, B2, x)
predicted = get_predictions(A2)[0]

st.image(x.reshape(28, 28), width=280, caption="Selected MNIST Digit", clamp=True)

if predicted == actual:
    st.success(f"✅ Correct Prediction: {predicted}")
else:
    st.error(f"❌ Incorrect Prediction: {predicted} (Actual: {actual})")

A2_all = forward_propagation(W1, B1, W2, B2, X_val)
val_predictions = get_predictions(A2_all)
val_accuracy = get_accuracy(val_predictions, y_val)
st.markdown(f"**Validation Accuracy:** {val_accuracy * 100:.2f}%")
