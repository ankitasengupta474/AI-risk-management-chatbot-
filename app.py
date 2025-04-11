import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np

# Load environment variables (GROQ API key)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB Client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="project_risks")

# Initialize Groq client
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    st.error("GROQ_API_KEY is not set. Please add it in Streamlit Secrets!")
    st.stop()

# Streamlit UI
st.title("AI Risk Management Chatbot")
st.write("Upload your document and start asking questions!")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "pdf", "txt", "docx"])

if uploaded_file is not None:
    file_text = uploaded_file.read().decode("utf-8", errors="ignore")

    # Embed and store the file text
    embedding = embedding_model.encode(file_text)
    collection.add(
        documents=[file_text],
        embeddings=[embedding.tolist()],
        metadatas=[{"source": uploaded_file.name}],
        ids=[uploaded_file.name]
    )
    st.success("Document embedded successfully!")

# User Question
question = st.text_input("Ask your question:")

if st.button("Submit"):
    if not question:
        st.warning("Please enter a question.")
    else:
        # Embed question
        question_embedding = embedding_model.encode(question)

        # Search similar docs
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=1
        )

        retrieved_text = results['documents'][0][0] if results['documents'] else "No relevant document found."

        # Send to GROQ
        prompt = f"Use the following document context to answer:

Context:
{retrieved_text}

Question: {question}
Answer:"

        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
        )

        answer = completion.choices[0].message.content
        st.subheader("Answer:")
        st.write(answer)
