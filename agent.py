from typing import Dict, Any, TypedDict
import logging
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from llm_config import get_groq_model, has_groq_api_key
from rag_engine import RAGEngine

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class AdvisorState(TypedDict):
    property_details: Dict[str, Any]
    predicted_price: float
    rag_context: str
    comps: list[dict[str, Any]]
    final_advice: str

class PropertyAdvisorAgent:
    def __init__(self):
        self.rag_engine = RAGEngine()
        
        # Ensure FAISS index exists before the agent tries to use it.
        # This will silently build or skip if it's already built.
        try:
            if not self.rag_engine.is_ready:
                self.rag_engine.build_index()
        except Exception as e:
            logging.warning(f"Could not initialize RAG index: {e}")
            
        # Use Groq as the primary LLM provider when the API key is available.
        if has_groq_api_key():
            self.llm = ChatGroq(model=get_groq_model(), temperature=0.2)
        else:
            self.llm = None
            logging.warning("No GROQ_API_KEY set. The agent will run in fallback mode.")

        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the LangGraph reasoning workflow."""
        workflow = StateGraph(AdvisorState)

        # Define the nodes (actions)
        workflow.add_node("retrieve_market_context", self._retrieve_market_context)
        workflow.add_node("retrieve_comps", self._retrieve_comps)
        workflow.add_node("generate_investment_advice", self._generate_investment_advice)

        # Define the edges (flow)
        workflow.set_entry_point("retrieve_market_context")
        workflow.add_edge("retrieve_market_context", "retrieve_comps")
        workflow.add_edge("retrieve_comps", "generate_investment_advice")
        workflow.add_edge("generate_investment_advice", END)

        return workflow.compile()

    def _retrieve_market_context(self, state: AdvisorState) -> Dict:
        """Retrieves knowledge from FAISS based on the property features."""
        details = state["property_details"]
        
        # Formulate a query based on the interesting features of this specific property
        query_parts = ["What are the investment trends and value considerations for a property with "]
        if details.get("basement") == "Yes":
            query_parts.append("a basement")
        if details.get("airconditioning") == "Yes":
            query_parts.append("air conditioning")
        if details.get("mainroad") == "Yes":
            query_parts.append("main road access")
            
        search_query = ", ".join(query_parts) + "?"
        
        logging.info(f"RAG Query: {search_query}")
        
        try:
            # If Groq is not configured, the RAG engine returns raw retrieved context.
            context = self.rag_engine.query(search_query, top_k=2)
        except Exception as e:
            logging.warning(f"RAG retrieval failed: {e}")
            context = "No specific market trends available at the moment."

        return {"rag_context": context}

    def _retrieve_comps(self, state: AdvisorState) -> Dict:
        """Retrieves the top 3 most similar comparable property sales."""
        details = state["property_details"]
        
        logging.info(f"Retrieving comparable properties for: {details}")
        
        try:
            comps = self.rag_engine.retrieve_comps(details, top_k=3)
            logging.info(f"Found {len(comps)} comparable properties")
        except Exception as e:
            logging.warning(f"Comps retrieval failed: {e}")
            comps = []

        return {"comps": comps}

    def _generate_investment_advice(self, state: AdvisorState) -> Dict:
        """Generates the final advice reasoning over the RAG context, comps, and ML prediction."""
        if not self.llm:
            # Build mock advice with comps
            comps_summary = ""
            if state.get("comps"):
                comps_summary = "\n\nComparable Properties:\n"
                for comp in state["comps"][:3]:
                    comps_summary += f"- {comp.get('location', 'N/A')}: ₹{comp.get('price', 'N/A')} ({comp.get('date', 'N/A')})\n"
            
            fallback_advice = (
                f"**[FALLBACK ADVICE - NO GROQ KEY]**\n"
                f"Based on the ML prediction of ₹{state['predicted_price']:,.0f}, this property is solid. "
                f"We also retrieved the following context from our knowledge base:\n"
                f"...\n{state['rag_context']}\n..."
                f"{comps_summary}"
            )
            return {"final_advice": fallback_advice}

        # Build comps context
        comps_context = ""
        if state.get("comps"):
            comps_context = "\n\nComparable Recent Sales:\n"
            for comp in state["comps"][:3]:
                location = comp.get('location', 'N/A')
                price = comp.get('price', 'N/A')
                date = comp.get('date', 'N/A')
                area = comp.get('area', 'N/A')
                comps_context += f"- {location}: ₹{price} ({area} sq ft, {date})\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior real estate investment advisor.
You are helping a client evaluate a property.
You will be provided with:
1. The property details.
2. The estimated ML price prediction.
3. Relevant market trends and local regulations (RAG Context).
4. Comparable recent property sales (Comps).

Write a concise, professional 3-paragraph investment summary.
Paragraph 1: Summarize the property and state the estimated price clearly.
Paragraph 2: Reference the comparable properties and how they validate or inform the pricing.
Paragraph 3: Give a final recommendation (Buy, Hold, or Pass) with reference to market comps."""),
            ("human", """Property Details: {details}
ML Estimated Price: ₹{price}

RAG Context / Market Rules:
{context}

Comparable Properties:
{comps}""")
        ])

        chain = prompt | self.llm
        
        response = chain.invoke({
            "details": str(state["property_details"]),
            "price": f"{state['predicted_price']:,.0f}",
            "context": state["rag_context"],
            "comps": comps_context
        })
        
        return {"final_advice": response.content}

    def run(self, property_details: Dict[str, Any], predicted_price: float) -> tuple[str, list]:
        """Kicks off the advisory agent and returns (advice, comps)."""
        initial_state = {
            "property_details": property_details,
            "predicted_price": predicted_price,
            "rag_context": "",
            "comps": [],
            "final_advice": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        return final_state["final_advice"], final_state.get("comps", [])

if __name__ == "__main__":
    # Example test run
    sample_property = {
        "area": 4500,
        "bedrooms": 3,
        "bathrooms": 2,
        "basement": "Yes",
        "airconditioning": "Yes",
        "mainroad": "No"
    }
    sample_price = 8500000.0  # ₹8.5M
    
    agent = PropertyAdvisorAgent()
    print("\n✅ Initializing Agentic Advisor...")
    advice, comps = agent.run(sample_property, sample_price)
    
    print("\n🤖 Final Agent Advice:")
    print("="*40)
    print(advice)
    print("="*40)
    
    if comps:
        print("\n📍 Comparable Properties Found:")
        for i, comp in enumerate(comps, 1):
            print(f"   {i}. {comp.get('location', 'N/A')} - ₹{comp.get('price', 'N/A')} ({comp.get('date', 'N/A')})")
        print("="*40)
