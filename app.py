#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import joblib


# In[ ]:


file_path="/kaggle/input/project-risk-management/it_project_risk_management_dataset (1).csv"

df = pd.read_csv(file_path)



# In[ ]:


features = [
    "total_budget", "current_spend", "resource_utilization", "delay_days", "budget_variance",
    "customer_payments_received", "tech_sector_performance", "market_volatility_index",
    "company_stock_performance", "technology_complexity", "integration_challenges",
    "team_turnover_rate", "regulatory_changes", "data_privacy_compliance"
]
target = "overall_risk_score"


# In[ ]:


X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initializing the model
rf = RandomForestRegressor(random_state=42)

# Randomized search with cross-validation
random_search = RandomizedSearchCV(
    estimator=rf, param_distributions=param_grid,
    n_iter=20, cv=5, verbose=1, n_jobs=-1, random_state=42
)


# In[ ]:


# Fitting the model
random_search.fit(X_train_scaled, y_train)

# Best model
best_model = random_search.best_estimator_


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


y_pred = best_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model RMSE: {rmse}")
print(f"Model R² Score: {r2}")

# Saving the optimized model and scaler in Kaggle output directory
optimized_model_path = "/kaggle/working/optimized_risk_prediction_model.pkl"
scaler_path = "/kaggle/working/scaler.pkl"


# In[ ]:


joblib.dump(best_model, optimized_model_path)
joblib.dump(scaler, scaler_path)

print(f"Optimized model saved at: {optimized_model_path}")
print(f"Scaler saved at: {scaler_path}")


# In[ ]:


import joblib

def load_model():
    return joblib.load("/kaggle/working/optimized_risk_prediction_model.pkl")

model = load_model()


# In[ ]:


get_ipython().system('pip install chromadb sentence-transformers')


# In[ ]:


pip install -U langchain-huggingface


# In[ ]:


import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize ChromaDB Persistent Storage
chroma_client = chromadb.PersistentClient(path="./chromadb_store")

# Load a local embedding model (no API key required)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a ChromaDB collection
collection = chroma_client.get_or_create_collection(name="project_risks")

# Create project descriptions based on the existing columns
df['project_description'] = df.apply(
    lambda row: f"{row['project_name']} in the {row['industry']} industry, managed by {row['company']}.", axis=1
)

# Convert project descriptions into vector embeddings
df["embeddings"] = df["project_description"].apply(lambda x: embedding_model.encode(x).tolist())
for idx, row in df.iterrows():
    collection.add(
        ids=[str(idx)],  # IDs must be strings
        embeddings=[row["embeddings"]],  # List of embeddings (each embedding must be a list)
        metadatas=[{
            "project_id": row["project_id"],
            "risk_score": row["overall_risk_score"]
        }]
    )

print("Embeddings stored successfully!")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()  # Machine learning model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # NLP model


# In[ ]:


def query_risk_analysis(query_text, top_k=3):
    query_embedding = embedding_model.encode(query_text).tolist()  # Use correct model
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    print("\nTop Risk Factors:", results)


# In[ ]:


def query_risk_analysis(query_text, top_k=3):
    """
    Given a project description or risk-related query, 
    this function retrieves similar risks and calculates an overall risk score.
    """
    # Generate embedding for the query text
    query_embedding = embedding_model.encode(query_text).tolist()

    # Query the vector database for relevant risk factors
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Extract risk scores (assuming results contain risk severity scores)
    if "documents" in results and results["documents"]:
        risk_factors = results["documents"][0]  # Extract the top results

        # Extract risk scores if available
        risk_scores = [item.get("score", 0) for item in risk_factors]  

        # Calculate the average risk score
        avg_risk = np.mean(risk_scores) if risk_scores else 0  
        
        # Define risk categories based on score
        if avg_risk >= 0.7:
            risk_level = "High"
        elif avg_risk >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Print the results
        print(f"\nTop Risk Factors: {risk_factors}")
        print(f"Overall Risk Score: {avg_risk:.2f} ({risk_level})")

        return {
            "risk_factors": risk_factors,
            "overall_risk": avg_risk,
            "risk_level": risk_level
        }
    else:
        print("\nNo relevant risk factors found in the database.")
        return {"risk_factors": [], "overall_risk": 0, "risk_level": "Unknown"}


# In[ ]:


def query_risk_analysis(query_text, top_k=3):
    """
    Given a project description or risk-related query, 
    this function retrieves similar risks and calculates an overall risk score.
    """
    # Generate embedding for the query text
    query_embedding = embedding_model.encode(query_text).tolist()

    # Query the vector database for relevant risk factors
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Extract risk scores (assuming results contain risk severity scores)
    if "documents" in results and results["documents"]:
        risk_factors = results["documents"][0]  # Extract the top results

        # Extract risk scores if available
        risk_scores = [item.get("score", 0) for item in risk_factors]  

        # Calculate the average risk score
        avg_risk = np.mean(risk_scores) if risk_scores else 0  
        
        # Define risk categories based on score
        if avg_risk >= 0.7:
            risk_level = "High"
        elif avg_risk >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Print the results
        print(f"\nTop Risk Factors: {risk_factors}")
        print(f"Overall Risk Score: {avg_risk:.2f} ({risk_level})")

        return {
            "risk_factors": risk_factors,
            "overall_risk": avg_risk,
            "risk_level": risk_level
        }
    else:
        print("\nNo relevant risk factors found in the database.")
        return {"risk_factors": [], "overall_risk": 0, "risk_level": "Unknown"}


# In[ ]:


# Import necessary libraries(REAL)
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb  # Use ChromaDB or Pinecone based on your setup

# Load a pre-trained NLP model for text embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # You can change the model

# Initialize the vector database (ChromaDB as an example)
client = chromadb.PersistentClient(path="risk_analysis_db")  # Ensure path exists
collection = client.get_or_create_collection(name="project_risks")

def query_risk_analysis(query_text, top_k=3):
    """
    Given a project description or risk-related query, 
    this function retrieves similar risks and calculates an overall risk score.
    """
    # Generate embedding for the query text
    query_embedding = embedding_model.encode(query_text).tolist()

    # Query the vector database for relevant risk factors
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Extract risk scores (assuming results contain risk severity scores)
    if "documents" in results and results["documents"]:
        risk_factors = results["documents"][0]  # Extract the top results

        # Extract risk scores if available
        risk_scores = [item.get("score", 0) for item in risk_factors]  

        # Calculate the average risk score
        avg_risk = np.mean(risk_scores) if risk_scores else 0  
        
        # Define risk categories based on score
        if avg_risk >= 0.7:
            risk_level = "High"
        elif avg_risk >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Print the results
        print(f"\nTop Risk Factors: {risk_factors}")
        print(f"Overall Risk Score: {avg_risk:.2f} ({risk_level})")

        return {
            "risk_factors": risk_factors,
            "overall_risk": avg_risk,
            "risk_level": risk_level
        }
    else:
        print("\nNo relevant risk factors found in the database.")
        return {"risk_factors": [], "overall_risk": 0, "risk_level": "Unknown"}

# Example Query
query_text = "What are the risks for an AI-based cloud computing project?"
query_risk_analysis(query_text)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()  # Machine learning model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # NLP model


# In[ ]:


def query_risk_analysis(query_text, top_k=3):
    query_embedding = embedding_model.encode(query_text).tolist()
    
    # Retrieve results
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    # DEBUG: Print raw results
    print("\nRaw Query Results:", results)
    
    # Extract relevant documents
    if "documents" in results and results["documents"]:
        risk_factors = results["documents"][0]
        print("\nExtracted Risk Factors:", risk_factors)

        risk_scores = [item.get("score", 0) for item in risk_factors]  
        avg_risk = np.mean(risk_scores) if risk_scores else 0  

        if avg_risk >= 0.7:
            risk_level = "High"
        elif avg_risk >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        print(f"\nOverall Risk Score: {avg_risk:.2f} ({risk_level})")
        return {"risk_factors": risk_factors, "overall_risk": avg_risk, "risk_level": risk_level}
    
    print("\nNo relevant risk factors found in the database.")
    return {"risk_factors": [], "overall_risk": 0, "risk_level": "Low"}


# In[ ]:


query_text = "What are the risks for an AI-based healthcare project?"
query_risk_analysis(query_text)


# In[ ]:


import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB Persistent Storage
chroma_client = chromadb.PersistentClient(path="./chromadb_store")

# Load a local embedding model (no API key required)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a ChromaDB collection
collection = chroma_client.get_or_create_collection(name="project_risks")

# Convert project descriptions into vector embeddings
df["embeddings"] = df["project_description"].apply(lambda x: embedding_model.encode(x).tolist())

# Store embeddings in ChromaDB
for idx, row in df.iterrows():
    collection.add(
        ids=[str(idx)],
        embeddings=[row["embeddings"]],
        metadatas=[{"project_id": row["project_id"], "risk_score": row["overall_risk_score"]}]
    )

print("Embeddings stored successfully!")

# Perform similarity search
query_text = "High budget variance and technology complexity"
query_embedding = embedding_model.encode(query_text).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("Top similar projects:", results)


# In[ ]:


import numpy as np

def query_risk_analysis(query_text, top_k=5):
    # Generate query embedding
    query_embedding = embedding_model.encode(query_text).tolist()
    
    # Retrieve similar projects from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # DEBUG: Print raw results
    print("\nRaw Query Results:", results)

    # Extract risk scores from retrieved projects
    risk_scores = []
    if "metadatas" in results and results["metadatas"]:
        for metadata in results["metadatas"][0]:  
            risk_scores.append(metadata.get("risk_score", 0))  

    # If no risk scores are found, return Low risk
    if not risk_scores:
        print("\nNo relevant risk factors found.")
        return {"risk_factors": [], "overall_risk": 0, "risk_level": "Low"}
    
    # Calculate the overall risk score
    avg_risk = np.mean(risk_scores)
    
    # Assign risk level
    if avg_risk >= 0.7:
        risk_level = "High"
    elif avg_risk >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    print(f"\nTop Risk Factors: {risk_scores}")
    print(f"Overall Risk Score: {avg_risk:.2f} ({risk_level})")

    return {"risk_factors": risk_scores, "overall_risk": avg_risk, "risk_level": risk_level}

# Example Query
query_text = "High budget variance and technology complexity"
query_risk_analysis(query_text)


# In[ ]:


import numpy as np

def query_risk_analysis(query_text, top_k=5):
    query_embedding = embedding_model.encode(query_text).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    risk_scores = [
        metadata.get("risk_score", 0) 
        for metadata in results.get("metadatas", [[]])[0] 
        if metadata
    ]

    if not risk_scores:
        return {"risk_factors": [], "overall_risk": 0, "risk_level": "Low"}
    
    avg_risk = np.mean(risk_scores)

    risk_level = "High" if avg_risk >= 0.7 else "Medium" if avg_risk >= 0.4 else "Low"

    return {"risk_factors": risk_scores, "overall_risk": avg_risk, "risk_level": risk_level}

# Example Query
query_text = "High budget variance and technology complexity"
result = query_risk_analysis(query_text)
print(result)


# In[ ]:


get_ipython().system('pip install torch')
get_ipython().system('pip install fastapi')
get_ipython().system('pip install streamlit')
get_ipython().system('pip install uvicorn')
get_ipython().system('pip install requests')


# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')


# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install datasets')



# In[ ]:


get_ipython().system('pip install groq')


# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GROQ_API")


# In[ ]:


pip install transformers torch chromadb sentence-transformers accelerate


# In[ ]:


# --- Load Groq API Key from Kaggle Secrets and inject into environment ---

from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
secret_value = user_secrets.get_secret("GROQ_API")
os.environ["GROQ_API_KEY"] = secret_value

print("✅ Groq API Key loaded successfully from Kaggle Secrets.")


# In[ ]:


import datetime
import logging
import numpy as np
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict, Any
import os  # Added for environment variable option

# Check if tkinter is available (for GUI mode)
HAS_TKINTER = False
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, font
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    # Test if display is available
    root = tk.Tk()
    root.destroy()
    HAS_TKINTER = True
except Exception:
    # Running in environment without display (like Kaggle)
    HAS_TKINTER = False

class ProjectKnowledgeBase:
    """Contains information about project challenges and mitigation strategies."""
    
    def __init__(self):
        """Initialize with project-specific information."""
        self.project_challenges = {
            "budget": {
                "description": "The project is experiencing budget overruns due to unexpected material cost increases and extended timelines.",
                "mitigation": "Implement stricter cost control measures, negotiate with vendors for better rates, and review the project scope to identify non-essential features that could be deprioritized."
            },
            "timeline": {
                "description": "The project is behind schedule due to resource constraints and technical integration challenges.",
                "mitigation": "Revise the project schedule, add additional experienced resources to critical path tasks, and implement daily stand-up meetings to quickly address blockers."
            },
            "resources": {
                "description": "Key team members have limited availability due to competing priorities and unexpected turnover.",
                "mitigation": "Cross-train team members, document knowledge to reduce dependency on specific individuals, and establish clear escalation paths for resource conflicts."
            },
            "technical": {
                "description": "Integration with legacy systems is proving more complex than anticipated, causing delays and quality issues.",
                "mitigation": "Bring in technical specialists familiar with the legacy systems, create a dedicated integration testing environment, and implement detailed error logging for faster troubleshooting."
            },
            "quality": {
                "description": "User acceptance testing has identified several critical defects that require significant rework.",
                "mitigation": "Implement more rigorous code reviews, add automated testing for regression issues, and conduct earlier stakeholder reviews to identify problems sooner."
            },
            "stakeholder": {
                "description": "Stakeholders have changing requirements and expectations, creating scope creep and confusion.",
                "mitigation": "Implement formal change management processes, conduct regular stakeholder alignment sessions, and maintain a prioritized backlog of requirements."
            },
            "market": {
                "description": "Market conditions are shifting, potentially impacting the project's business case and priorities.",
                "mitigation": "Conduct regular market analysis, maintain flexibility in the implementation approach, and keep the business case updated to reflect changing conditions."
            }
        }
        
        # General project information
        self.project_info = {
            "name": "Enterprise Digital Transformation Initiative",
            "objective": "Modernize legacy systems and implement new digital capabilities to improve operational efficiency and customer experience.",
            "timeline": "12 months (Currently in month 7)",
            "budget": "$2.8M",
            "team_size": "15 core team members plus extended stakeholders",
            "key_milestones": [
                "Requirements & Design: Months 1-3 (Completed)",
                "Development Phase 1: Months 3-6 (Completed)",
                "Development Phase 2: Months 6-9 (In Progress - Currently Delayed)",
                "Testing & Integration: Months 8-11 (Not Started)",
                "Deployment & Handover: Months 11-12 (Not Started)"
            ]
        }
    
    def get_project_info(self):
        """Returns general project information."""
        return self.project_info
    
    def get_challenge_info(self, challenge_type):
        """Returns information about a specific challenge type."""
        challenge_type = challenge_type.lower()
        for key, info in self.project_challenges.items():
            if challenge_type in key:
                return {key: info}
        return None
    
    def get_all_challenges(self):
        """Returns all project challenges."""
        return self.project_challenges
    
    def search_challenges(self, query):
        """Searches for challenges related to the query."""
        query = query.lower()
        relevant_challenges = {}
        
        # Map common terms to challenge categories
        keyword_mapping = {
            "cost": "budget", "expense": "budget", "money": "budget", "finance": "budget",
            "schedule": "timeline", "deadline": "timeline", "late": "timeline", "delay": "timeline",
            "staff": "resources", "team": "resources", "personnel": "resources", "hiring": "resources",
            "system": "technical", "integration": "technical", "code": "technical", "architecture": "technical",
            "defect": "quality", "bug": "quality", "testing": "quality", "standards": "quality",
            "client": "stakeholder", "management": "stakeholder", "scope": "stakeholder",
            "competition": "market", "industry": "market", "economic": "market"
        }
        
        # Check for direct challenge types first
        for challenge in self.project_challenges.keys():
            if challenge in query:
                relevant_challenges[challenge] = self.project_challenges[challenge]
        
        # If no direct matches, check for related terms
        if not relevant_challenges:
            for keyword, challenge in keyword_mapping.items():
                if keyword in query and challenge not in relevant_challenges:
                    relevant_challenges[challenge] = self.project_challenges[challenge]
        
        return relevant_challenges if relevant_challenges else self.project_challenges

class ProjectRiskAnalysisChatbot:
    def __init__(
        self, 
        groq_api_key=None,  # You can provide your API key here when initializing
        model_name="llama3-8b-8192", 
        embedding_model="all-MiniLM-L6-v2",
        use_gui=None  # None means auto-detect
    ):
        """
        Initialize the Risk Analysis Chatbot using Groq AI
        
        Args:
            groq_api_key (str): Your Groq API key
            model_name (str): Groq model identifier
            embedding_model (str): Sentence Transformer for embeddings
            use_gui (bool): Whether to use GUI. If None, auto-detects.
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Determine if GUI should be used
        if use_gui is None:
            self.use_gui = HAS_TKINTER
        else:
            self.use_gui = use_gui and HAS_TKINTER
            
        if self.use_gui:
            self.logger.info("Running in GUI mode")
        else:
            self.logger.info("Running in CLI mode")

        try:
            # Initialize ChromaDB for vector storage
            self.chroma_client = chromadb.PersistentClient(path="./risk_analysis_memory")
            self.collection = self.chroma_client.get_or_create_collection(
                name="project_risks"
            )

            # Load embedding model
            self.embedding_model = SentenceTransformer(embedding_model)

            # Initialize Groq client
            self.groq_client = Groq(api_key=groq_api_key)
            self.model_name = model_name

            # Initialize risk analysis agents
            self.market_agent = MarketAnalysisAgent()
            self.scoring_agent = RiskScoringAgent(self.collection, self.embedding_model)
            self.tracking_agent = ProjectStatusTrackingAgent()
            self.reporting_agent = ReportingAgent()
            
            # Add knowledge base
            self.knowledge_base = ProjectKnowledgeBase()
            
            # Initialize UI
            self.root = None
            self.recent_report = None

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise
    
    def is_risk_related(self, query_text):
        """
        Check if a query is related to project risks, challenges, or management
        
        Args:
            query_text (str): User's query
            
        Returns:
            bool: True if query is related to project information, False otherwise
        """
        project_keywords = [
            # Original risk keywords
            "risk", "project", "budget", "cost", "deadline", "delay", "timeline",
            "resource", "team", "technical", "quality", "market", "financial",
            "competitor", "competition", "challenge", "issue", "problem", "concern",
            "schedule", "staff", "technology", "system", "requirement",
            
            # Added more general project keywords
            "task", "objective", "goal", "milestone", "deliverable", "status",
            "progress", "mitigation", "strategy", "solve", "solution", "fix",
            "handle", "address", "manage", "approach", "stakeholder", "sponsor",
            "scope", "feature", "implementation", "plan", "phase", "testing",
            "deployment", "release", "development", "design", "integration",
            "what", "how", "why", "when", "who"  # General question words
        ]
        
        query_lower = query_text.lower()
        return "challenge" in query_lower or any(keyword in query_lower for keyword in project_keywords)


    def analyze_project_risks(self, query_text):
        """
        Comprehensive project risk analysis
        
        Args:
            query_text (str): User's risk query
        
        Returns:
            dict: Detailed risk analysis report
        """
        try:
            # Perform multi-agent risk analysis
            market_risk = self.market_agent.analyze_market_risk(query_text)
            risk_data = self.scoring_agent.query_risk_analysis(query_text)
            status_risk = self.tracking_agent.track_project_status()

            # Get risk names and values
            risk_names = []
            risk_values = []
            
            # Market risks
            market_names = ["Market Volatility", "Financial Trends", "Industry Competition"]
            for i, score in enumerate(market_risk.get("risk_factors", [])):
                if i < len(market_names):
                    risk_names.append(market_names[i])
                    risk_values.append(score)
            
            # Internal risks
            internal_names = ["Budget Issues", "Timeline Delays", "Resource Constraints", 
                              "Technical Challenges", "Quality Issues"]
            for i, score in enumerate(risk_data.get("risk_factors", [])):
                if i < len(internal_names):
                    risk_names.append(internal_names[i])
                    risk_values.append(score)
                    
            # Status risks
            status_names = ["Project Delays", "Team Member Changes"]
            for i, score in enumerate(status_risk.get("risk_factors", [])):
                if i < len(status_names):
                    risk_names.append(status_names[i])
                    risk_values.append(score)

            # Combine all risk factors
            all_risks = (
                risk_data.get("risk_factors", []) + 
                market_risk.get("risk_factors", []) + 
                status_risk.get("risk_factors", [])
            )

            # Calculate overall risk
            overall_risk = np.mean(all_risks) if all_risks else 0
            risk_level = (
                "High" if overall_risk >= 0.7 
                else "Medium" if overall_risk >= 0.4 
                else "Low"
            )

            # Generate detailed report
            report = {
                "query": query_text,
                "risk_factors": all_risks,
                "risk_names": risk_names,
                "risk_values": risk_values,
                "overall_risk": overall_risk,
                "risk_level": risk_level,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Generate AI-enhanced explanation with the query text
            report['explanation'] = self.generate_risk_explanation(report, query_text)

            # Save the latest report
            self.recent_report = report
            
            return report

        except Exception as e:
            self.logger.error(f"Risk analysis error: {e}")
            return {"error": str(e)}

    def generate_risk_explanation(self, report, query_text=None):
        """
        Generate natural language explanation of risk report using Groq AI
        with focus on actual project challenges and mitigation strategies
        
        Args:
            report (dict): Risk analysis report
            query_text (str): Original user query
        
        Returns:
            str: AI-generated risk explanation
        """
        try:
            if query_text is None:
                query_text = report.get('query', '')
            
            # Determine if this is about challenges, mitigations, or general risk
            query_lower = query_text.lower()
            is_about_challenges = any(word in query_lower for word in ['challenge', 'problem', 'issue', 'concern'])
            is_about_mitigation = any(word in query_lower for word in ['mitigate', 'solve', 'address', 'solution', 'fix', 'handle'])
            
            # Get relevant challenge information
            relevant_challenges = self.knowledge_base.search_challenges(query_lower)
            challenge_info = ""
            for challenge_type, info in relevant_challenges.items():
                challenge_info += f"- {challenge_type.title()}: {info['description']}\n"
                if is_about_mitigation:
                    challenge_info += f"  Mitigation: {info['mitigation']}\n"
            
            # Get risk names and scores for the prompt
            risk_details = ""
            if 'risk_names' in report and 'risk_values' in report:
                for name, value in zip(report['risk_names'], report['risk_values']):
                    risk_level = "HIGH" if value >= 0.7 else "MEDIUM" if value >= 0.4 else "LOW"
                    risk_details += f"- {name}: {value:.2f} ({risk_level})\n"
            
            # Get project information
            project_info = self.knowledge_base.get_project_info()
            project_summary = f"Project: {project_info['name']}\nObjective: {project_info['objective']}\nTimeline: {project_info['timeline']}\nBudget: {project_info['budget']}\n"
            
            # Prepare appropriate prompt based on query type
            if is_about_challenges and not is_about_mitigation:
                prompt = f"""
                The user asked about project challenges: "{query_text}"
                
                Based on the following information, provide a detailed explanation of the relevant project challenges:
                
                {project_summary}
                
                Risk Assessment:
                - Overall Risk Level: {report['risk_level']} ({report['overall_risk']:.2f})
                
                Relevant Project Challenges:
                {challenge_info}
                
                Focus on explaining the challenges in detail. Include specific issues the project is facing and why they matter. 
                Be specific and practical in your explanation.
                """
            elif is_about_mitigation:
                prompt = f"""
                The user asked about mitigating project challenges: "{query_text}"
                
                Based on the following information, provide detailed mitigation strategies for the relevant challenges:
                
                {project_summary}
                
                Risk Assessment:
                - Overall Risk Level: {report['risk_level']} ({report['overall_risk']:.2f})
                
                Relevant Project Challenges and Mitigation Strategies:
                {challenge_info}
                
                Focus on explaining the mitigation strategies in detail. Include specific actions that can be taken to address each challenge.
                Provide practical and actionable advice.
                """
            else:
                prompt = f"""
                The user asked: "{query_text}"
                
                Provide a business-focused explanation of the following project and its risks:
                
                {project_summary}
                
                Risk Assessment:
                - Overall Risk Level: {report['risk_level']} ({report['overall_risk']:.2f})
                - Key Risk Factors:
                {risk_details}
                
                Relevant Project Challenges:
                {challenge_info}
                
                Explain the business implications of these risks and challenges. Focus on what these risk scores mean 
                for the project in practical terms. Keep the explanation clear, concise, and directly relevant to 
                business stakeholders.
                """

            # Check if API key is available
            if not self.groq_client.api_key:
                self.logger.error("No Groq API key provided")
                return "Error: No API key available. Please provide a valid Groq API key."

            # Generate explanation using Groq
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500  # Increased token count for more detailed responses
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Explanation generation error: {e}")
            return f"Unable to generate detailed explanation. Error: {str(e)}"

    def generate_risk_chart(self, report=None, save_path=None):
        """
        Generate bar chart visualization of risk factors
        
        Args:
            report (dict, optional): Risk report to visualize. Uses recent report if None.
            save_path (str, optional): Path to save the chart image, for CLI mode
            
        Returns:
            matplotlib figure or None
        """
        if report is None:
            report = self.recent_report
            
        if not report or 'risk_names' not in report or 'risk_values' not in report:
            return None
            
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get risk data
            risk_names = report['risk_names']
            risk_values = report['risk_values']
            
            # Create color map based on risk levels
            colors = ['#91cf60' if v < 0.4 else '#fee08b' if v < 0.7 else '#fc8d59' for v in risk_values]
            
            # Create horizontal bar chart
            bars = ax.barh(risk_names, risk_values, color=colors)
            
            # Add a vertical line at thresholds
            ax.axvline(x=0.4, color='#fee08b', linestyle='--', alpha=0.7)
            ax.axvline(x=0.7, color='#fc8d59', linestyle='--', alpha=0.7)
            
            # Add value labels to the bars
            for bar in bars:
                width = bar.get_width()
                ax.text(max(width + 0.02, 0.05), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', 
                        va='center')
            
            # Set title and labels
            ax.set_title('Project Risk Factors Analysis')
            ax.set_xlabel('Risk Score (0-1)')
            ax.set_xlim(0, 1.0)
            
            # Add legend for risk levels
            import matplotlib.patches as mpatches
            low_patch = mpatches.Patch(color='#91cf60', label='Low Risk (<0.4)')
            medium_patch = mpatches.Patch(color='#fee08b', label='Medium Risk (0.4-0.7)')
            high_patch = mpatches.Patch(color='#fc8d59', label='High Risk (>0.7)')
            ax.legend(handles=[low_patch, medium_patch, high_patch], loc='lower right')
            
            # Make layout tight
            plt.tight_layout()
            
            # Save the chart if path provided (for CLI mode)
            if save_path:
                plt.savefig(save_path)
                plt.close(fig)
                return save_path
                
            # Return figure for use in UI
            return fig
            
        except Exception as e:
            self.logger.error(f"Chart generation error: {e}")
            return None

    def show_explanation_window(self):
        """Shows a window with the detailed risk explanation"""
        if not self.use_gui or not self.recent_report or 'explanation' not in self.recent_report:
            return
            
        explanation = self.recent_report['explanation']
        
        # Create new window
        explanation_window = tk.Toplevel(self.root)
        explanation_window.title("Risk Explanation")
        explanation_window.geometry("600x400")
        
        # Add explanation text
        text_area = scrolledtext.ScrolledText(explanation_window, wrap=tk.WORD, font=("Helvetica", 12))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_area.insert(tk.END, explanation)
        text_area.config(state=tk.DISABLED)  # Make read-only

    def show_barchart_window(self):
        """Shows a window with the risk barchart"""
        if not self.use_gui or not self.recent_report:
            return
            
        # Generate chart
        fig = self.generate_risk_chart()
        if not fig:
            return
            
        # Create new window
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Risk Analysis Chart")
        chart_window.geometry("800x600")
        
        # Create canvas for matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def start_gui(self):
        """Start the GUI chat interface"""
        if not self.use_gui:
            self.logger.warning("GUI mode not available, falling back to CLI")
            self._cli_chat()
            return
            
        # Create main window
        self.root = tk.Tk()
        self.root.title("Project Risk Analysis Chatbot")
        self.root.geometry("800x600")
        
        # Configure styles
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Subtitle.TLabel", font=("Helvetica", 12))
        style.configure("Risk.TLabel", font=("Helvetica", 14, "bold"))
        
        # Top frame for title
        top_frame = ttk.Frame(self.root, padding="10 10 10 0")
        top_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(top_frame, text="🚨 Project Risk Analysis Chatbot", style="Title.TLabel")
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(top_frame, 
                                   text="Ask about project challenges, budget variances, or specific risk concerns.", 
                                   style="Subtitle.TLabel")
        subtitle_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Middle frame for chat history
        chat_frame = ttk.Frame(self.root, padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED)
        chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame for input and buttons
        input_frame = ttk.Frame(self.root, padding=10)
        input_frame.pack(fill=tk.X)
        
        user_input = ttk.Entry(input_frame, font=("Helvetica", 11))
        user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Function to handle sending messages
        def send_message():
            query = user_input.get().strip()
            if not query:
                return
                
            # Display user message
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, "\n\nYou: " + query + "\n")
            
            # Clear input field
            user_input.delete(0, tk.END)
            
            # Check for "explain" command
            if query.lower() == "explain":
                if self.recent_report and 'explanation' in self.recent_report:
                    chat_display.insert(tk.END, "\n💡 Detailed Explanation:\n")
                    chat_display.insert(tk.END, self.recent_report['explanation'] + "\n")
                else:
                    chat_display.insert(tk.END, "\nNo recent risk analysis to explain. Please ask about a project risk first.\n")
                chat_display.see(tk.END)
                chat_display.config(state=tk.DISABLED)
                return
                
            # Check if query is risk-related
            if not self.is_risk_related(query):
                chat_display.insert(tk.END, "\nSorry, I can't answer that. Please ask about project risks.\n")
                chat_display.see(tk.END)
                chat_display.config(state=tk.DISABLED)
                return
            
            # Process the query
            try:
                # Show thinking message
                chat_display.insert(tk.END, "\n🤔 Analyzing risks... Please wait.\n")
                chat_display.see(tk.END)
                chat_display.update_idletasks()
                
                # Analyze risks
                risk_report = self.analyze_project_risks(query)
                
                # Remove thinking message
                chat_display.delete("end-2l", "end-1c")
                
                # Display risk level only (no brief explanation)
                chat_display.insert(tk.END, f"\n🔍 Risk Analysis Results:\n")
                chat_display.insert(tk.END, f"Risk Level: {risk_report['risk_level']} (Score: {risk_report['overall_risk']:.2f})\n\n")
                
                # Create button frame for this response
                chat_display.insert(tk.END, "Type 'explain' for a detailed explanation or use these buttons:\n")
                
                # Create button for explanation
                explanation_btn = ttk.Button(
                    chat_display, 
                    text="View Detailed Explanation", 
                    command=self.show_explanation_window
                )
                chat_display.window_create(tk.END, window=explanation_btn)
                
                # Create button for barchart
                chat_display.insert(tk.END, "   ")
                barchart_btn = ttk.Button(
                    chat_display, 
                    text="View Risk Chart", 
                    command=self.show_barchart_window
                )
                chat_display.window_create(tk.END, window=barchart_btn)
                
                chat_display.insert(tk.END, "\n")
                
            except Exception as e:
                chat_display.insert(tk.END, f"\nError: {str(e)}\n")
                
            # Scroll to the end and disable editing
            chat_display.see(tk.END)
            chat_display.config(state=tk.DISABLED)
        
        # Button to send message
        send_button = ttk.Button(input_frame, text="Send", command=send_message)
        send_button.pack(side=tk.RIGHT)
        
        # Bind Enter key to send message
        self.root.bind('<Return>', lambda event: send_message())
        
        # Welcome message
        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, "🚨 Risk Analysis Chatbot: Hello! I can help you analyze project risks.\n")
        chat_display.insert(tk.END, "Ask about project challenges, budget variances, or specific risk concerns.\n")
        chat_display.insert(tk.END, "Type 'explain' after risk analysis to see detailed explanation.\n")
        chat_display.config(state=tk.DISABLED)
        
        # Set focus to the input field
        user_input.focus()
        
        # Start the GUI
        self.root.mainloop()

    def _cli_chat(self):
        """
        Interactive risk analysis chat through command line
        """
        print("🚨 Risk Analysis Chatbot: Hello! I can help you analyze project risks.")
        print("Ask about project challenges, budget variances, or specific risk concerns.")
        print("Type 'exit' to end the conversation.")
        print("Type 'explain' for the latest detailed explanation.")
        print("Type 'chart' to display risk factors (as text in CLI mode).")
        print("Type 'save_chart <filename.png>' to save the risk chart as an image.")
        print("Type 'help' to see example queries and keywords.")
        print("\n📌 Example queries:")
        print("- What are the risks in this project?")
        print("- How can we mitigate technical issues?")
        print("- What are the challenges with AI in healthcare?")
        print("- What is the budget risk?")
        print("- Show me the chart")
        print("- Explain")
        print("\n🔑 Keywords you can try:")
        print("  budget, timeline, resources, technical, quality, stakeholder, market, competition, delay, problem, fix")
    
        while True:
            try:
                user_input = input("\nYou: ").strip()
    
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("🚨 Risk Analysis Chatbot: Goodbye! Stay risk-aware.")
                    break
    
                if user_input.lower() == 'explain':
                    if self.recent_report and 'explanation' in self.recent_report:
                        print("\n💡 Detailed Explanation:")
                        print(self.recent_report.get('explanation', 'No detailed explanation available.'))
                    else:
                        print("\nNo recent risk analysis to explain. Please ask about a project risk first.")
                    continue
    
                if user_input.lower() == 'chart':
                    if self.recent_report and 'risk_names' in self.recent_report:
                        print("\n📊 Risk Factors Chart (Text Version):")
                        if 'risk_names' in self.recent_report and 'risk_values' in self.recent_report:
                            max_name_len = max(len(name) for name in self.recent_report['risk_names'])
                            for name, value in zip(self.recent_report['risk_names'], self.recent_report['risk_values']):
                                bar = '█' * int(value * 20)
                                risk_level = "HIGH" if value >= 0.7 else "MED" if value >= 0.4 else "LOW"
                                print(f"{name.ljust(max_name_len)} | {bar} {value:.2f} ({risk_level})")
                    else:
                        print("\nNo recent risk analysis to display. Please ask about a project risk first.")
                    continue
    
                if user_input.lower().startswith('save_chart '):
                    if self.recent_report and 'risk_names' in self.recent_report:
                        filename = user_input[11:].strip()
                        if not filename.endswith(('.png', '.jpg', '.pdf')):
                            filename += '.png'
                        try:
                            saved_path = self.generate_risk_chart(save_path=filename)
                            if saved_path:
                                print(f"Chart saved to {saved_path}")
                            else:
                                print("Failed to save chart")
                        except Exception as e:
                            print(f"Error saving chart: {e}")
                    else:
                        print("\nNo recent risk analysis to save. Please ask about a project risk first.")
                    continue
    
                if not self.is_risk_related(user_input):
                    print("\nSorry, I can't answer that. Please ask about project risks.")
                    continue
    
                print("\n🤔 Analyzing risks... Please wait.")
                risk_report = self.analyze_project_risks(user_input)
    
                print(f"\n🔍 Risk Analysis Results:")
                print(f"Risk Level: {risk_report['risk_level']} (Score: {risk_report['overall_risk']:.2f})")
    
                explanation = risk_report.get('explanation', '')
                if explanation:
                    print("\n💡 Explanation:")
                    print(explanation)
                else:
                    print("\n(No detailed explanation available.)")
    
                print("\nType 'chart' to see visual risk breakdown or 'save_chart filename.png' to export.")
    
            except Exception as e:
                print(f"\nError: {str(e)}")

    def chat(self):
        """
        Start the chatbot interface - dispatches to CLI or GUI based on settings
        """
        if self.use_gui:
            self.start_gui()
        else:
            self._cli_chat()

class MarketAnalysisAgent:
    """Agent that analyzes market-related risks for the project."""
    
    def __init__(self):
        self.market_data = {
            "market_volatility": 0.65,
            "financial_trends": 0.48,
            "industry_competition": 0.72
        }
    
    def analyze_market_risk(self, query_text):
        """
        Analyze market risks based on the query
        
        Args:
            query_text (str): User's query
            
        Returns:
            dict: Market risk analysis
        """
        query_lower = query_text.lower()
        
        # Adjust baseline risk factors based on query
        volatility = self.market_data["market_volatility"]
        financial = self.market_data["financial_trends"]
        competition = self.market_data["industry_competition"]
        
        # Adjust based on query content
        if "market" in query_lower or "industry" in query_lower:
            volatility += 0.05
            competition += 0.05
        
        if "finance" in query_lower or "budget" in query_lower or "cost" in query_lower:
            financial += 0.1
            
        if "competitor" in query_lower or "competition" in query_lower:
            competition += 0.15
            
        # Normalize scores to 0-1 range
        volatility = min(max(volatility, 0), 1)
        financial = min(max(financial, 0), 1)
        competition = min(max(competition, 0), 1)
        
        # Calculate overall market risk
        overall_risk = np.mean([volatility, financial, competition])
        
        # Return risk analysis
        return {
            "risk_factors": [volatility, financial, competition],
            "overall_market_risk": overall_risk
        }

class RiskScoringAgent:
    """Agent that scores and stores project risks."""
    
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        self.risk_baseline = {
            "budget": 0.7,
            "timeline": 0.65,
            "resources": 0.55,
            "technical": 0.6,
            "quality": 0.5
        }
    
    def query_risk_analysis(self, query_text):
        """
        Query and analyze risks based on the query text
        
        Args:
            query_text (str): User's query
            
        Returns:
            dict: Risk analysis results
        """
        query_lower = query_text.lower()
        
        # Adjust baseline risk factors based on query
        budget_risk = self.risk_baseline["budget"]
        timeline_risk = self.risk_baseline["timeline"]
        resource_risk = self.risk_baseline["resources"]
        technical_risk = self.risk_baseline["technical"]
        quality_risk = self.risk_baseline["quality"]
        
        # Adjust based on query content
        if "budget" in query_lower or "cost" in query_lower or "finance" in query_lower:
            budget_risk += 0.1
            
        if "timeline" in query_lower or "schedule" in query_lower or "deadline" in query_lower:
            timeline_risk += 0.1
            
        if "resource" in query_lower or "staff" in query_lower or "team" in query_lower:
            resource_risk += 0.1
            
        if "technical" in query_lower or "technology" in query_lower or "system" in query_lower:
            technical_risk += 0.1
            
        if "quality" in query_lower or "defect" in query_lower or "bug" in query_lower:
            quality_risk += 0.1
            
        # Normalize scores to 0-1 range
        budget_risk = min(max(budget_risk, 0), 1)
        timeline_risk = min(max(timeline_risk, 0), 1)
        resource_risk = min(max(resource_risk, 0), 1)
        technical_risk = min(max(technical_risk, 0), 1)
        quality_risk = min(max(quality_risk, 0), 1)
        
        # Calculate overall risk
        overall_risk = np.mean([budget_risk, timeline_risk, resource_risk, technical_risk, quality_risk])
        
        # Store the query and analysis in the collection
        if query_text:
            embedding = self.embedding_model.encode(query_text).tolist()
            self.collection.add(
                embeddings=[embedding],
                documents=[query_text],
                metadatas=[{
                    "overall_risk": float(overall_risk),
                    "budget_risk": float(budget_risk),
                    "timeline_risk": float(timeline_risk),
                    "resource_risk": float(resource_risk),
                    "technical_risk": float(technical_risk),
                    "quality_risk": float(quality_risk),
                    "timestamp": datetime.datetime.now().isoformat()
                }],
                ids=[f"query_{datetime.datetime.now().timestamp()}"]
            )
        
        # Return risk analysis
        return {
            "risk_factors": [budget_risk, timeline_risk, resource_risk, technical_risk, quality_risk],
            "overall_risk": overall_risk
        }

class ProjectStatusTrackingAgent:
    """Agent that tracks project status and related risks."""
    
    def __init__(self):
        self.project_status = {
            "current_phase": "Development Phase 2",
            "timeline_status": "Delayed",
            "budget_status": "Over Budget",
            "resource_status": "Understaffed"
        }
    
    def track_project_status(self):
        """
        Track project status and evaluate status-related risks
        
        Returns:
            dict: Project status risk analysis
        """
        # Calculate risk factors based on project status
        timeline_risk = 0.7 if self.project_status["timeline_status"] == "Delayed" else 0.3
        resource_risk = 0.8 if self.project_status["resource_status"] == "Understaffed" else 0.4
        
        # Calculate overall status risk
        overall_risk = np.mean([timeline_risk, resource_risk])
        
        # Return status risk analysis
        return {
            "risk_factors": [timeline_risk, resource_risk],
            "overall_status_risk": overall_risk,
            "status": self.project_status
        }

class ReportingAgent:
    """Agent that generates risk reports and visualizations."""
    
    def __init__(self):
        pass
    
    def generate_risk_report(self, risk_data):
        """
        Generate a comprehensive risk report
        
        Args:
            risk_data (dict): Risk analysis data
            
        Returns:
            str: Formatted risk report
        """
        # Generate formatted risk report
        report = f"Project Risk Analysis Report\n"
        report += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += f"Overall Risk Level: {risk_data.get('risk_level', 'Unknown')}\n"
        report += f"Overall Risk Score: {risk_data.get('overall_risk', 0):.2f}\n\n"
        
        report += "Risk Factors:\n"
        for i, factor in enumerate(risk_data.get('risk_factors', [])):
            factor_name = f"Factor {i+1}"
            if 'risk_names' in risk_data and i < len(risk_data['risk_names']):
                factor_name = risk_data['risk_names'][i]
            report += f"- {factor_name}: {factor:.2f}\n"
        
        return report

def main():
    """Main function to run the Project Risk Analysis Chatbot."""
    # Try to get API key from environment variable
    api_key = os.environ.get("GROQ_API_KEY", None)
    
    # Initialize and run the chatbot
    chatbot = ProjectRiskAnalysisChatbot(
        groq_api_key=api_key,
        use_gui=None  # Auto-detect
    )
    
    # Start chat interface
    chatbot.chat()

if __name__ == "__main__":
    main()


# In[ ]:


get_ipython().system('du -sh /kaggle/working')


# In[ ]:


get_ipython().system('df -h')


# In[ ]:


get_ipython().system('free -h')


# In[ ]:




