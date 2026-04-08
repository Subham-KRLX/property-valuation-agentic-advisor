# Intelligent Property Valuation Agentic Advisor 🏠

A hybrid AI/ML project for real-estate valuation that combines a Random Forest price predictor with retrieval-augmented market context and Groq-powered investment guidance.

## 🔗 Live Demo
**Access the live application here:** [Live Demo](https://property-valuation-agentic-advisor-xvfy6pzq5caq72fmxlzrak.streamlit.app/)

## 🎯 Project Overview
The core objective is to estimate property value from historical housing data and then explain that estimate with grounded, retrieval-backed investment advice.

### Key Features
- **Price Prediction**: Interactive form to input property details and get instant value estimates.
- **Groq-Powered Advisory**: Generates concise investment guidance from property features, model output, and retrieved market context.
- **Model Insights**: Comparative analysis and performance metrics (R², MAE, RMSE).
- **Feature Importance**: Visual breakdown of which factors most influence property pricing.
- **RAG Knowledge Layer**: Uses FAISS and local knowledge-base documents to ground the AI response in retrieved information.

## 🛠️ Technology Stack
- **Languages**: Python 3.13+
- **Machine Learning**: `scikit-learn` (Random Forest, Linear Regression)
- **Data Processing**: `pandas`, `numpy`
- **LLM Provider**: `Groq` via `langchain-groq`
- **Orchestration**: `LangChain`, `LangGraph`, `FAISS`
- **Visualization**: `matplotlib`, `seaborn` (for metadata generation), `streamlit` (native charts)
- **Deployment**: `Streamlit Community Cloud`

## 📊 Model Performance
The system was trained on the [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).

| Metric | Random Forest | Linear Regression |
| :--- | :--- | :--- |
| **R-squared (R²)** | 0.582 | 0.627 |
| **MAE** | ₹1,080,958 | ₹999,836 |

## 🚀 Local Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/NssGourav/property-valuation-agentic-advisor.git
   cd property-valuation-agentic-advisor
   ```

2. **Environment Setup**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   ```bash
   export GROQ_API_KEY="your_groq_api_key"
   export GROQ_MODEL="llama-3.1-8b-instant"  # Optional
   ```

4. **Train the Model (Optional)**
   ```bash
   python3 train_model.py
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

If `GROQ_API_KEY` is not set, the app still runs. The valuation flow remains available, and the advisory layer falls back to a clearly labeled non-Groq response.

## 📂 Project Structure
- `app.py`: Streamlit frontend and application logic.
- `train_model.py`: Data preprocessing and model training pipeline.
- `agent.py`: LangGraph advisory workflow powered by Groq.
- `rag_engine.py`: FAISS-backed retrieval layer for market context.
- `llm_config.py`: Shared Groq provider configuration helpers.
- `models/`: Contains the serialized `house_model.pkl`.
- `assets/`: Contains `model_metadata.json` and static assets.
- `data/`: Local storage for the dataset (excluded from Git).

## 👥 Team
- **Nss Gourav**
- **Subham Sangwan**

---
*Developed for Project 9: Intelligent Property Price Prediction.*
