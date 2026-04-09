import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from validator import PropertyInputValidator

from agent import PropertyAdvisorAgent
from pdf_report import build_property_report

@st.cache_resource
def load_advisor_agent():
    return PropertyAdvisorAgent()

PAGE_TITLE = "Intelligent Property Valuation"
PAGE_ICON = "🏠"
MODEL_PATH = Path("models/house_model.pkl")
METADATA_PATH = Path("assets/model_metadata.json")

NUMERICAL_FEATURES = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
BINARY_FEATURES = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES

class ValuationApp:
    def __init__(self):
        self.model = self._load_model()
        
    def _load_model(self):
        if not MODEL_PATH.exists():
            st.error(f"⚠️ Model file not found at {MODEL_PATH}. Please run `train_model.py` first.")
            return None
        
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None

    def _load_metadata(self):
        if not METADATA_PATH.exists():
            return None

        try:
            with open(METADATA_PATH, "r") as file:
                return json.load(file)
        except Exception:
            return None

    def render_header(self):
        st.title(f"{PAGE_ICON} {PAGE_TITLE}")
        st.markdown("---")

    def render_form(self):
        if self.model is None:
            return

        with st.form("valuation_form"):
            st.subheader("Property Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=3000, step=100)
                bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
                bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=1)
                stories = st.number_input("Stories", min_value=1, max_value=4, value=2)
                parking = st.number_input("Parking Spots", min_value=0, max_value=3, value=1)

            with col2:
                mainroad = st.selectbox("Main Road Access", ["Yes", "No"])
                guestroom = st.selectbox("Guest Room", ["Yes", "No"])
                basement = st.selectbox("Basement", ["Yes", "No"])
                hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
                airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])

            submitted = st.form_submit_button("🔮 Predict Price")
            
            if submitted:
                self._predict_price(
                    area, bedrooms, bathrooms, stories, parking,
                    mainroad, guestroom, basement, hotwaterheating, airconditioning
                )

    def _predict_price(self, area, bedrooms, bathrooms, stories, parking,
                      mainroad, guestroom, basement, hotwaterheating, airconditioning):

        # Inputs for validation
        raw_inputs = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
        }

        # Validate inputs
        validator = PropertyInputValidator()
        validation = validator.validate(raw_inputs)

        if validation.warnings:
            st.markdown("---")
            st.markdown("#### ⚠️ Input Warnings")
            for warn in validation.warnings:
                st.warning(warn)

        # Block prediction on errors
        if not validation.is_valid:
            st.markdown("---")
            st.markdown("#### ❌ Validation Errors")
            for err in validation.errors:
                st.error(err)
            st.info("Please correct the details and try again.")
            return

        # Build feature dataframe
        input_data = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": 1 if mainroad == "Yes" else 0,
            "guestroom": 1 if guestroom == "Yes" else 0,
            "basement": 1 if basement == "Yes" else 0,
            "hotwaterheating": 1 if hotwaterheating == "Yes" else 0,
            "airconditioning": 1 if airconditioning == "Yes" else 0,
            "parking": parking,
        }

        features = [
            "area", "bedrooms", "bathrooms", "stories", "mainroad",
            "guestroom", "basement", "hotwaterheating", "airconditioning", "parking"
        ]

        input_df = pd.DataFrame([input_data], columns=features)

        # Run prediction
        try:
            prediction = self.model.predict(input_df)[0]
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return

        st.markdown("---")
        st.success(f"### 💰 Estimated Property Value: ₹{prediction:,.0f}")

        st.info("**Property Summary:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Area", f"{area:,} sq ft")
            st.metric("Bedrooms", bedrooms)
        with col_b:
            st.metric("Bathrooms", bathrooms)
            st.metric("Stories", stories)
        with col_c:
            st.metric("Parking", parking)
            amenities = sum(
                1 for val in [mainroad, guestroom, basement, hotwaterheating, airconditioning]
                if val == "Yes"
            )
            st.metric("Amenities", f"{amenities}/5")

        st.balloons()

        # --- Agentic Advisory Layer ---
        st.markdown("---")
        st.subheader("🤖 AI Investment Advisor")
        agent_input = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
        }

        advice = "Advisory summary was unavailable at the time of export."
        comps = []
        try:
            advisor = load_advisor_agent()
            with st.spinner("Analyzing market trends and regulations..."):
                advice, comps = advisor.run(agent_input, prediction)

            st.success("**Investment Summary & Recommendation**")
            st.write(advice)
            
            # Display comps if available
            if comps:
                st.markdown("---")
                st.subheader("📍 Comparable Properties (Comps)")
                comp_cols = st.columns(len(comps[:3]))
                for idx, comp in enumerate(comps[:3]):
                    with comp_cols[idx]:
                        st.info(
                            f"**{comp.get('location') or 'N/A'}**\n\n"
                            f"₹{comp.get('price') or 'N/A'}\n"
                            f"({comp.get('date') or 'N/A'})"
                        )
        except Exception as e:
            advice = f"Advisory unavailable: {e}"
            st.warning("The valuation was generated, but the AI advisory could not be completed.")
            st.caption(str(e))

        st.markdown("---")
        st.subheader("📄 Export Report")
        metadata = self._load_metadata()
        report_details = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "amenities": f"{amenities}/5",
        }
        report_bytes = build_property_report(
            property_details=report_details,
            estimated_price=prediction,
            advisory_text=advice,
            validation_warnings=validation.warnings,
            metadata=metadata,
            comps=comps,
        )
        file_name = f"valuation-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
        st.caption("Download a polished PDF summary of the valuation, advisory, and key model metrics.")
        st.download_button(
            label="Download PDF Report",
            data=report_bytes,
            file_name=file_name,
            mime="application/pdf",
            use_container_width=True,
        )

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    
    app = ValuationApp()
    
    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Model Insights"])
    
    with tab1:
        app.render_header()
        app.render_form()
        
    with tab2:
        st.header("Model Performance & Insights")
        metadata = app._load_metadata()
        if metadata:
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            metrics = metadata["metrics"]
            m1.metric("R² Score", f"{metrics['r2']:.3f}")
            m2.metric("MAE", f"₹{metrics['mae']:,.0f}")
            m3.metric("RMSE", f"₹{metrics['rmse']:,.0f}")
            
            st.markdown("---")
            st.subheader("Feature Importance")
            st.info("This chart shows which property features most significantly influence the price prediction.")
            
            importance_df = pd.DataFrame({
                'Feature': list(metadata["feature_importance"].keys()),
                'Importance': list(metadata["feature_importance"].values())
            }).sort_values(by='Importance', ascending=True)
            
            st.bar_chart(data=importance_df, x='Feature', y='Importance', horizontal=True)
            
            st.markdown("---")
            st.subheader("Methodology")
            st.write("""
            The model uses a **Random Forest Regressor**, a robust machine learning algorithm that builds multiple 
            decision trees and merges them together to get a more accurate and stable prediction.
            
            **Key Features Analyzed:**
            - **Physical attributes:** Area, Bedrooms, Bathrooms, Stories.
            - **Infrastructural features:** Main road access, Basement, Air conditioning.
            - **Property status:** Parking available, Guest room, Hot water heating.
            """)
        else:
            st.warning("Model metadata not found. Please run `train_model.py` to generate insights.")
main()
