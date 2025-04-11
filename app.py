import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variable (GROQ API Key)
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Initialize Groq client
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    st.error("GROQ_API_KEY not found!")

# Streamlit UI
st.title("AI Risk Management Chatbot")

input_data = st.text_input("Enter your project description:")

if st.button("Analyze Risk"):
    if input_data:
        # Embed input
        embedding = embedding_model.encode([input_data])

        # Scale embedding
        scaled_embedding = scaler.transform(embedding)

        # Predict risk
        risk_score = model.predict(scaled_embedding)[0]

        st.metric(label="Predicted Risk Score", value=f"{risk_score:.2f}")

        # Ask Groq for risk explanation
        prompt = f"What are the potential risks for a project described as: {input_data}?"
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}]
        )
        ai_message = response.choices[0].message.content
        st.subheader("AI-Generated Risk Explanation:")
        st.write(ai_message)
    else:
        st.warning("Please enter some project description.")
