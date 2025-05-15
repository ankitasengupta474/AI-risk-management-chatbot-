Project Title: AI Risk Management Chatbot

Description:  
This is a lightweight AI-based chatbot built using Streamlit. It is designed to help analyze project risk data from CSV files. The app uses semantic search to understand the uploaded content and responds to user queries using the LLaMA3 model from Groq. It combines vector storage, LLM-based reasoning, and a simple user interface to make project data exploration interactive and intelligent.

What It Can Do:
- Accept CSV uploads related to project risks or other structured datasets
- Convert each row into semantic embeddings using sentence transformers
- Store and search content using ChromaDB for fast retrieval
- Allow users to ask questions and receive accurate, contextual answers
- Include a placeholder for future predictive analytics using RandomForest

Technologies Used:
- Streamlit for the user interface
- SentenceTransformers for generating semantic embeddings
- ChromaDB for storing and retrieving embedded data
- Groq API (LLaMA3 model) for generating natural language responses
- scikit-learn for modeling and scaling (optional feature)
- joblib, numpy, pandas, matplotlib for data handling and utilities

How to Run Locally:
1. Clone the repository
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

2. Install the required Python packages
   pip install -r requirements.txt

3. Set the GROQ API key  
   Option 1: Create a file at .streamlit/secrets.toml with the following content:
   GROQ_API_KEY = "your_groq_api_key_here"

   Option 2: Export the key in your terminal
   export GROQ_API_KEY=your_groq_api_key_here

4. Run the app
   streamlit run app.py

Dev Container Support:
This project includes configuration for GitHub Codespaces or VS Code Dev Containers.  
Just open the folder in VS Code and choose "Reopen in Container".  
The app will automatically start and be available at port 8501.

File Structure:
- app.py — main application logic
- requirements.txt — list of Python dependencies
- .devcontainer/devcontainer.json — container setup for development
- .streamlit/secrets.toml — for storing API key (optional)
- README.md — project overview

CSV Format Example:
Make sure your CSV is structured like this:

Risk ID, Description, Impact, Probability  
1, Delay in procurement, High, 0.7  
2, Resource unavailability, Medium, 0.5  

Each row will be semantically indexed and used for answering questions.

Contact:
For support or inquiries, please email: info@citizenone.in
