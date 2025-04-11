import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np

# Load environment variables (GROQ API key)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
collection = chroma_client.get_or_create_collection(name="project_risks")

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
        query_embedding = embedding_model.encode(query_text).tolist()

        # Search ChromaDB
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        # Process Results
        risk_scores = []
        if "metadatas" in results and results["metadatas"]:
            for metadata in results["metadatas"][0]:
                risk_scores.append(metadata.get("risk_score", 0))

        if risk_scores:
            avg_risk = np.mean(risk_scores)
            risk_level = "High" if avg_risk >= 0.7 else "Medium" if avg_risk >= 0.4 else "Low"

            st.subheader("🔍 Risk Analysis Result")
            st.write(f"**Average Risk Score:** {avg_risk:.2f}")
            st.write(f"**Risk Level:** {risk_level}")
        else:
            avg_risk = 0
            risk_level = "Low"
            st.info("No matching risks found in database. Risk assumed to be Low.")

        # Generate explanation using Groq
        if groq_client:
            with st.spinner("Generating AI-based risk explanation..."):
                prompt = f"Explain in business language why the following project might be {risk_level} risk: {query_text}"
                try:
                    response = groq_client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=400
                    )
                    explanation = response.choices[0].message.content
                    st.success("AI Explanation:")
                    st.write(explanation)
                except Exception as e:
                    st.error(f"Groq API Error: {e}")
        else:
            st.warning("Groq API key not found. Cannot generate detailed explanation.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, ChromaDB, Sentence-Transformers, and Groq AI.")
