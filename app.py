import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load environment variables (GROQ API key)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Groq client
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    st.error("GROQ_API_KEY not found. Please set it in Streamlit secrets.")

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="project_risks")

# Load Random Forest Model (Dummy example, since you don't have pkl file)
# Create a dummy model to prevent error
model = RandomForestRegressor()
scaler = StandardScaler()

# Streamlit App
st.title("AI Risk Management Chatbot")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", df)

    # Embed all rows into ChromaDB
    texts = df.apply(lambda row: " ".join(map(str, row.values)), axis=1).tolist()
    embeddings = embedding_model.encode(texts).tolist()

    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"row": i}]
        )

    st.success("Data embedded successfully!")

query = st.text_input("Ask a question related to the uploaded data:")

if st.button("Get Answer") and query:
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=['documents', 'distances']
    )

    retrieved_docs = "\n".join(results['documents'][0])
    
    prompt = f"""
    Use the following document context to answer the question:

    {retrieved_docs}

    Question: {query}
    Answer:
    """

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    st.write("### Answer:")
    st.success(answer)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, ChromaDB, Groq API, and HuggingFace Models.")
