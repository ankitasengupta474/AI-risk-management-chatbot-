import streamlit as st
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables (GROQ API key)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Groq client
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

# Streamlit App
st.set_page_config(page_title="AI Risk Management Chatbot", page_icon="🚨")
st.title("🚨 Project Risk Management Chatbot")
st.markdown("Enter your project description to analyze potential risks.")

query_text = st.text_area("Describe your project or ask about risks:", height=150)

if st.button("Analyze Risk"):
    if not query_text:
        st.warning("Please enter a project description.")
    else:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(query_text)

        # Basic risk scoring (mock logic without ChromaDB)
        risk_score = np.random.uniform(0.3, 0.8)  # Random risk score for demonstration
        risk_level = "High" if risk_score >= 0.7 else "Medium" if risk_score >= 0.4 else "Low"

        st.subheader("🔍 Risk Analysis Result")
        st.write(f"**Predicted Risk Score:** {risk_score:.2f}")
        st.write(f"**Risk Level:** {risk_level}")

        # Generate explanation using Groq
        if groq_client:
            with st.spinner("Generating AI-based risk explanation..."):
                prompt = f"Explain the project risks based on the description: '{query_text}' and assigned risk level: {risk_level}. Focus on why it may face risks and practical advice."
                try:
                    response = groq_client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=400
                    )
                    explanation = response.choices[0].message.content
                    st.success("AI Risk Explanation:")
                    st.write(explanation)
                except Exception as e:
                    st.error(f"Groq API Error: {e}")
        else:
            st.warning("Groq API key not found. Cannot generate detailed explanation.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, Sentence-Transformers, and Groq AI.")
