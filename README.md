# Intelligent Property Valuation Agentic Advisor 🏠

A hybrid AI/ML project for real-estate valuation that combines a Random Forest price predictor with retrieval-augmented market context (RAG) and Groq-powered investment guidance. This project turns raw housing data into actionable investment summaries.

## 🔗 Live Demo
**Access the live application here:** [Live Demo](https://property-valuation-agentic-advisor-xvfy6pzq5caq72fmxlzrak.streamlit.app/)

## 🎯 Project Overview
The core objective is to estimate property value from historical housing data and then explain that estimate with grounded, retrieval-backed investment advice.

### Key Features
- **Intelligent Price Prediction**: Interactive form with real-time validation for physical property attributes.
- **Agentic Investment Advisor**: A LangGraph-powered advisor that reasons across ML predictions, market context, and comparable sales.
- **RAG Knowledge Layer**: Uses FAISS and local knowledge-base documents to ground AI responses in factual, retrieved information.
- **Comparable Sales Retrival**: Automatically finds the most similar recent property transactions from the knowledge base.
- **PDF Investment Brief**: Generates a professional, exportable report containing valuation, advisory, and model metrics.
- **Model Insights**: Detailed performance metrics (R², MAE, RMSE) and feature importance analysis.

## 🛠️ Technology Stack
- **Core**: Python 3.13+, Streamlit
- **ML/Data**: `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Agentic AI**: `LangGraph`, `LangChain`, `langchain-groq`
- **Vector DB**: `FAISS` (Facebook AI Similarity Search)
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Reporting**: `ReportLab` (PDF generation)

## 🚀 Quick Start (Local Setup)

### 1. Clone & Install
```bash
git clone https://github.com/NssGourav/property-valuation-agentic-advisor.git
cd property-valuation-agentic-advisor
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt # For testing
```

### 2. Configure Environment
Create a `.env` file or export the following variables:
| Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | API key from [Groq Console](https://console.groq.com/) | *Required for Advisory* |
| `GROQ_MODEL` | Model ID for advisory (e.g., `llama-3.1-8b-instant`) | `llama-3.1-8b-instant` |
| `KAGGLE_USERNAME` | Kaggle username for dataset download | *Optional* |
| `KAGGLE_KEY` | Kaggle API key | *Optional* |

### 3. Run the App
```bash
streamlit run app.py
```

## 🔄 Core Workflows

### Model Retraining
To rebuild the price prediction model from the Kaggle dataset:
```bash
python3 train_model.py
```
This updates `models/house_model.pkl` and writes structured training metadata to `assets/model_metadata.json`.

### Knowledge Base & RAG
The RAG system indexes documents in `data/knowledge_base/`. 
- **Comparable Sales**: Stored in `comparable_sales.txt`. The system parses this to find similar properties.
- **Market Trends**: Stored in `market_trends.txt`. Used to ground the AI advisory.
The FAISS index is built automatically on the first run of the app or agent.

### Testing
Run the unit test suite to verify core logic:
```bash
python3 -m pytest tests/
```

## 🛡️ Robustness & Fallbacks
The system is designed to be resilient to missing dependencies:
- **No Groq API Key**: The app still runs perfectly for valuation. The advisor provides a "Fallback Mode" summary using raw retrieved context without LLM generation.
- **No Kaggle Credentials**: If `Housing.csv` is missing and download fails, clear instructions are provided for manual download.
- **Missing FAISS Index**: The system attempts to rebuild the index if source documents are present.

## 📂 Project Structure
- `app.py`: Streamlit entry point.
- `train_model.py`: Data preprocessing, training, and evaluation pipeline.
- `agent.py`: LangGraph advisory workflow.
- `rag_engine.py`: Vector search and comp parsing logic.
- `validator.py`: Input guardrails for property features.
- `pdf_report.py`: ReportLab PDF engine.
- `llm_config.py`: Shared LLM provider configuration.
- `assets/`: Model metadata and static assets.
- `data/knowledge_base/`: Source text for RAG.
- `models/`: Serialized models and FAISS index.

## 👥 Team
- **Nss Gourav**
- **Subham Sangwan**

---
*Developed for Project 9: Intelligent Property Price Prediction.*
