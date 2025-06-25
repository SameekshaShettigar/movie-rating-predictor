import streamlit as st
import pickle
import numpy as np

# Page config and custom styling
st.set_page_config(page_title="🎬 Movie Rating Predictor", layout="centered")

# Inject CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1608369562760-e5d9b07936bb?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }
    h1 {
        text-align: center;
        font-size: 2.8em;
        color: #FFD700;
        text-shadow: 1px 1px 5px #000;
    }
    button[kind="primary"] {
        background-color: #FFD93D;
        color: black;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
try:
    model = pickle.load(open("model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Required files not found. Please run train_model.py first.")
    st.stop()

st.markdown("<h1>🎞️ Movie Rating Predictor 🎯</h1>", unsafe_allow_html=True)
st.markdown("### _Enter movie details below to get its predicted IMDb rating_ 🎥")

# Get dropdown options from encoders
genres = encoders["Genre"].classes_
directors = encoders["Director"].classes_
actor1s = encoders["Actor 1"].classes_
actor2s = encoders["Actor 2"].classes_

# Inputs
genre = st.selectbox("🎭 Genre", genres)
director = st.selectbox("🎬 Director", directors)
actor1 = st.selectbox("🧑‍🎤 Lead Actor 1", actor1s)
actor2 = st.selectbox("🧑‍🤝‍🧑 Lead Actor 2", actor2s)

# Button
if st.button("🚀 Predict Rating"):
    try:
        g = encoders["Genre"].transform([genre])[0]
        d = encoders["Director"].transform([director])[0]
        a1 = encoders["Actor 1"].transform([actor1])[0]
        a2 = encoders["Actor 2"].transform([actor2])[0]

        pred = model.predict([[g, d, a1, a2]])[0]
        st.success(f"🎯 Predicted IMDb Rating: **{pred:.2f}** ⭐")
    except Exception as e:
        st.error("⚠️ Error during prediction.")
        st.text(str(e))
