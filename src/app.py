import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
from src.validator import PropertyInputValidator

from src.agent import PropertyAdvisorAgent
from src.pdf_report import build_property_report

PAGE_TITLE = "Intelligent Property Valuation"
PAGE_ICON = "🏠"
MODEL_PATH = Path("models/house_model.pkl")
METADATA_PATH = Path("assets/model_metadata.json")

NUMERICAL_FEATURES = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
BINARY_FEATURES = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_global_styles() -> None:
    st.markdown(
        """
        <style>
          /* Page background + default typography */
          .stApp {
            background: radial-gradient(1200px 800px at 10% 0%, rgba(99, 102, 241, 0.14), transparent 55%),
                        radial-gradient(900px 700px at 90% 10%, rgba(16, 185, 129, 0.12), transparent 55%),
                        linear-gradient(180deg, rgba(2, 6, 23, 0.0) 0%, rgba(2, 6, 23, 0.03) 100%);
          }

          /* Reduce awkward top padding */
          div[data-testid="stAppViewContainer"] > .main {
            padding-top: 1.25rem;
          }

          /* Hero */
          .pv-hero {
            padding: 1.25rem 1.25rem 1.1rem 1.25rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.75), rgba(15, 23, 42, 0.55));
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.10);
            margin-bottom: 1rem;
          }
          .pv-hero-title {
            font-size: 1.75rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            line-height: 1.2;
          }
          .pv-hero-subtitle {
            margin-top: 0.35rem;
            color: rgba(148, 163, 184, 0.95);
            font-size: 0.95rem;
          }
          .pv-badges {
            margin-top: 0.7rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
          }
          .pv-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.82rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(2, 6, 23, 0.35);
          }

          /* Sidebar polish */
          section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(148, 163, 184, 0.18);
          }
          section[data-testid="stSidebar"] .stMarkdown {
            color: rgba(15, 23, 42, 0.95);
          }

          /* Buttons */
          button[kind="primary"] {
            border-radius: 12px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_inr(value: float) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return f"₹{value}"


@st.cache_resource
def load_advisor_agent():
    return PropertyAdvisorAgent()


class ValuationApp:
    def __init__(self):
        self.model = self._load_model()
        self.validator = PropertyInputValidator()
        self.metadata = self._load_metadata()
        
    def _load_model(self):
        if not MODEL_PATH.exists():
            st.error(f"### 🛑 Model Not Found")
            st.warning(
                f"The core valuation model was not found at `{MODEL_PATH}`. \n\n"
                "**To fix this:**\n"
                "Run the training script in your terminal:\n"
                "```bash\npython3 src/train_model.py\n```\n"
                "This will download the dataset and train the Random Forest model."
            )
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
        advisory_mode = "Template-based advisor"
        model_status = "Model loaded" if self.model is not None else "Model missing"

        st.markdown(
            f"""
            <div class="pv-hero">
              <div class="pv-hero-title">{PAGE_ICON} {PAGE_TITLE}</div>
              <div class="pv-hero-subtitle">
                ML valuation + RAG-grounded investment narrative — exportable as a PDF brief.
              </div>
              <div class="pv-badges">
                <div class="pv-badge">🧠 {model_status}</div>
                <div class="pv-badge">🤖 {advisory_mode}</div>
                <div class="pv-badge">📄 PDF export</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        st.sidebar.markdown("## Inputs")
        st.sidebar.caption("Enter property details, then run a valuation. Results appear in the main panel.")

        if self.model is None:
            st.sidebar.warning("Model missing — run `python3 src/train_model.py` first.")
            return
        st.sidebar.markdown("### Status")
        st.sidebar.caption(f"Model: {'loaded' if self.model is not None else 'missing'}")
        st.sidebar.caption("Advisor: Template-based")
        if self.metadata and "metrics" in self.metadata:
            metrics = self.metadata["metrics"]
            st.sidebar.caption(f"R²: {metrics.get('r2', 0):.3f} • MAE: {_format_inr(metrics.get('mae', 0))}")

        if "pv_last_result" not in st.session_state:
            st.session_state.pv_last_result = None

        with st.sidebar.form("valuation_form"):
            st.markdown("### Property details")
            area = st.number_input(
                "Area (sq ft)",
                min_value=300,
                max_value=25000,
                value=int(st.session_state.get("pv_area", 3000)),
                step=50,
                help="Built-up area in square feet.",
            )
            col1, col2 = st.columns(2)
            with col1:
                bedrooms = st.number_input(
                    "Bedrooms",
                    min_value=1,
                    max_value=10,
                    value=int(st.session_state.get("pv_bedrooms", 3)),
                    step=1,
                )
                bathrooms = st.number_input(
                    "Bathrooms",
                    min_value=1,
                    max_value=6,
                    value=int(st.session_state.get("pv_bathrooms", 2)),
                    step=1,
                )
            with col2:
                stories = st.number_input(
                    "Stories",
                    min_value=1,
                    max_value=5,
                    value=int(st.session_state.get("pv_stories", 2)),
                    step=1,
                )
                parking = st.number_input(
                    "Parking spots",
                    min_value=0,
                    max_value=5,
                    value=int(st.session_state.get("pv_parking", 1)),
                    step=1,
                )

            st.markdown("### Amenities")
            mainroad = st.checkbox("Main road access", value=bool(st.session_state.get("pv_mainroad", True)))
            guestroom = st.checkbox("Guest room", value=bool(st.session_state.get("pv_guestroom", False)))
            basement = st.checkbox("Basement", value=bool(st.session_state.get("pv_basement", False)))
            hotwaterheating = st.checkbox("Hot water heating", value=bool(st.session_state.get("pv_hotwaterheating", False)))
            airconditioning = st.checkbox("Air conditioning", value=bool(st.session_state.get("pv_airconditioning", True)))

            submitted = st.form_submit_button("Estimate value", use_container_width=True, type="primary")

        # Persist inputs for a smoother UX across reruns.
        st.session_state.pv_area = area
        st.session_state.pv_bedrooms = bedrooms
        st.session_state.pv_bathrooms = bathrooms
        st.session_state.pv_stories = stories
        st.session_state.pv_parking = parking
        st.session_state.pv_mainroad = mainroad
        st.session_state.pv_guestroom = guestroom
        st.session_state.pv_basement = basement
        st.session_state.pv_hotwaterheating = hotwaterheating
        st.session_state.pv_airconditioning = airconditioning

        if st.sidebar.button("Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("pv_"):
                    del st.session_state[key]
            st.rerun()

        if submitted:
            self._run_valuation(
                area=area,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                stories=stories,
                parking=parking,
                mainroad=mainroad,
                guestroom=guestroom,
                basement=basement,
                hotwaterheating=hotwaterheating,
                airconditioning=airconditioning,
            )

        last = st.session_state.get("pv_last_result")
        if last and last.get("status") == "ok":
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Latest result")
            st.sidebar.metric("Estimated value", _format_inr(last["prediction"]))
            if last["validation"].warnings:
                st.sidebar.caption(f"⚠️ {len(last['validation'].warnings)} warning(s) flagged by guardrails.")
            st.sidebar.caption("Tip: switch to the “Model Insights” tab for methodology and feature importance.")

    def render_main(self):
        if self.model is None:
            return
        last = st.session_state.get("pv_last_result")
        if not last:
            empty = st.container(border=True)
            with empty:
                st.subheader("Get started")
                st.write(
                    "Use the sidebar to enter property details, then click **Estimate value**. "
                    "You’ll get an ML-backed valuation, comparable context, and an investment summary."
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info("1) Input details\n\nArea, beds, baths, stories, parking, amenities.")
                with c2:
                    st.info("2) Run valuation\n\nGuardrails flag outliers before prediction.")
                with c3:
                    st.info("3) Export brief\n\nDownload a PDF you can share.")
            return

        self._render_result(last)

    def _run_valuation(
        self,
        *,
        area: float,
        bedrooms: int,
        bathrooms: int,
        stories: int,
        parking: int,
        mainroad: bool,
        guestroom: bool,
        basement: bool,
        hotwaterheating: bool,
        airconditioning: bool,
    ) -> None:

        # Inputs for validation
        raw_inputs = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
        }

        # Validate inputs
        validation = self.validator.validate(raw_inputs)

        if not validation.is_valid:
            st.session_state.pv_last_result = {
                "status": "invalid",
                "validation": validation,
            }
            return

        # Build feature dataframe
        input_data = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": 1 if mainroad else 0,
            "guestroom": 1 if guestroom else 0,
            "basement": 1 if basement else 0,
            "hotwaterheating": 1 if hotwaterheating else 0,
            "airconditioning": 1 if airconditioning else 0,
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
            st.session_state.pv_last_result = {
                "status": "error",
                "error": str(e),
                "validation": validation,
            }
            return

        amenities = sum([mainroad, guestroom, basement, hotwaterheating, airconditioning])

        # --- Agentic Advisory Layer ---
        agent_input = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": "Yes" if mainroad else "No",
            "guestroom": "Yes" if guestroom else "No",
            "basement": "Yes" if basement else "No",
            "hotwaterheating": "Yes" if hotwaterheating else "No",
            "airconditioning": "Yes" if airconditioning else "No",
            "parking": parking,
        }

        advice = "Advisory summary was unavailable at the time of export."
        comps = []
        advisory_error = None
        try:
            advisor = load_advisor_agent()
            with st.spinner("Analyzing market trends and comparable sales..."):
                advice, comps = advisor.run(agent_input, prediction)
        except Exception as e:
            advisory_error = str(e)
            advice = "Advisory unavailable at the moment. The valuation still completed successfully."

        st.session_state.pv_last_result = {
            "status": "ok",
            "prediction": float(prediction),
            "inputs": agent_input,
            "raw_inputs": {
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
            },
            "amenities": amenities,
            "validation": validation,
            "advice": advice,
            "comps": comps,
            "advisory_error": advisory_error,
            "generated_at": datetime.now(),
        }

    def _render_result(self, result: dict) -> None:
        status = result.get("status")
        if status == "invalid":
            st.divider()
            error_box = st.container(border=True)
            with error_box:
                st.markdown("#### ❌ Validation errors")
                for err in result["validation"].errors:
                    st.error(err, icon="🚨")
                if result["validation"].warnings:
                    st.markdown("#### ⚠️ Warnings")
                    for warn in result["validation"].warnings:
                        st.warning(warn, icon="⚠️")
                st.info("Fix the inputs in the sidebar and try again.")
            return

        if status == "error":
            st.divider()
            box = st.container(border=True)
            with box:
                st.error("Prediction failed.")
                st.caption(result.get("error", "Unknown error"))
            return

        if status != "ok":
            return

        prediction = result["prediction"]
        raw_inputs = result["raw_inputs"]
        validation = result["validation"]
        advice = result["advice"]
        comps = result.get("comps") or []
        generated_at = result.get("generated_at")

        st.divider()

        top = st.container(border=True)
        with top:
            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.markdown("### Estimated value")
                st.markdown(f"## {_format_inr(prediction)}")
                subtitle = "This is a model estimate from historical training data; treat as a directional baseline."
                if isinstance(generated_at, datetime):
                    subtitle = f"{subtitle} Generated {generated_at.strftime('%Y-%m-%d %H:%M')}."
                st.caption(subtitle)
            with c2:
                st.markdown("### Guardrails")
                if validation.warnings:
                    st.warning(f"{len(validation.warnings)} warning(s) flagged", icon="⚠️")
                    with st.expander("View warnings"):
                        for warn in validation.warnings:
                            st.write(f"- {warn}")
                else:
                    st.success("No anomalies detected", icon="✅")
                if self.metadata and "metrics" in self.metadata:
                    m = self.metadata["metrics"]
                    st.caption(
                        f"Model metrics: R² {m.get('r2', 0):.3f} • "
                        f"MAE {_format_inr(m.get('mae', 0))} • RMSE {_format_inr(m.get('rmse', 0))}"
                    )

        overview = st.container(border=True)
        with overview:
            st.markdown("### Property overview")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Area", f"{int(raw_inputs['area']):,} sq ft")
            col_b.metric("Bedrooms", int(raw_inputs["bedrooms"]))
            col_c.metric("Bathrooms", int(raw_inputs["bathrooms"]))
            col_d.metric("Stories", int(raw_inputs["stories"]))

            col_e, col_f, col_g = st.columns(3)
            col_e.metric("Parking", int(raw_inputs["parking"]))
            col_f.metric("Amenities", f"{int(result['amenities'])}/5")
            col_g.metric("Advisory", "Template")

        c_left, c_right = st.columns([1.15, 0.85], gap="large")
        with c_left:
            advice_card = st.container(border=True)
            with advice_card:
                st.markdown("### 🤖 AI investment advisor")
                st.write(advice)
                if result.get("advisory_error"):
                    with st.expander("View advisory error"):
                        st.caption(result["advisory_error"])

        with c_right:
            comps_card = st.container(border=True)
            with comps_card:
                st.markdown("### 📍 Comparable context")
                if not comps:
                    st.info("No comparable sales found for this run.")
                else:
                    top_comps = comps[:3]
                    cols = st.columns(len(top_comps))
                    for idx, comp in enumerate(top_comps):
                        with cols[idx]:
                            with st.container(border=True):
                                location = comp.get("location") or "N/A"
                                price = comp.get("price")
                                date = comp.get("date") or "N/A"
                                area = comp.get("area")
                                st.markdown(f"**{location}**")
                                st.caption(f"📅 {date}")
                                st.metric("Price", _format_inr(price) if price is not None else "N/A")
                                if area is not None:
                                    st.caption(f"Area: {area} sq ft")

                    with st.expander("View as table"):
                        comps_df = pd.DataFrame(comps)
                        if "price" in comps_df.columns:
                            comps_df["price"] = comps_df["price"].apply(_format_inr)
                        st.dataframe(comps_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📄 Property investment brief")
        metadata = self._load_metadata()
        report_details = {
            "area": raw_inputs["area"],
            "bedrooms": raw_inputs["bedrooms"],
            "bathrooms": raw_inputs["bathrooms"],
            "stories": raw_inputs["stories"],
            "parking": raw_inputs["parking"],
            "mainroad": "Yes" if raw_inputs["mainroad"] else "No",
            "guestroom": "Yes" if raw_inputs["guestroom"] else "No",
            "basement": "Yes" if raw_inputs["basement"] else "No",
            "hotwaterheating": "Yes" if raw_inputs["hotwaterheating"] else "No",
            "airconditioning": "Yes" if raw_inputs["airconditioning"] else "No",
            "amenities": f"{int(result['amenities'])}/5",
        }
        report_bytes = build_property_report(
            property_details=report_details,
            estimated_price=prediction,
            advisory_text=advice,
            validation_warnings=validation.warnings,
            metadata=metadata,
            comps=comps,
        )
        file_name = f"property-investment-brief-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
        st.caption("Download a PDF brief with valuation, comps, model metrics, and advisory.")
        st.download_button(
            label="Download PDF brief",
            data=report_bytes,
            file_name=file_name,
            mime="application/pdf",
            use_container_width=True,
        )

def main():
    _inject_global_styles()
    app = ValuationApp()
    
    app.render_sidebar()
    tab1, tab2 = st.tabs(["🏠 Valuation", "📊 Model Insights"])
    
    with tab1:
        app.render_header()
        app.render_main()
        
    with tab2:
        st.header("Model Performance & Insights")
        metadata = app._load_metadata()
        if metadata:
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            metrics = metadata["metrics"]
            m1.metric("R² Score", f"{metrics['r2']:.3f}")
            m2.metric("MAE", f"₹{metrics['mae']:,.0f}")
            m3.metric("RMSE", f"₹{metrics['rmse']:,.0f}")
            within_10 = metrics.get("within_10_pct")
            m4.metric("Within 10%", f"{within_10 * 100:.1f}%" if isinstance(within_10, (int, float)) else "N/A")

            if metadata.get("classification_metrics"):
                with st.expander("Accuracy / Precision / Recall (derived label)"):
                    cm = metadata["classification_metrics"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{cm.get('accuracy', 0) * 100:.1f}%")
                    c2.metric("Precision", f"{cm.get('precision', 0) * 100:.1f}%")
                    c3.metric("Recall", f"{cm.get('recall', 0) * 100:.1f}%")
                    c4.metric("F1", f"{cm.get('f1', 0) * 100:.1f}%")
                    st.caption(
                        "These are computed on a derived high-value label (price ≥ median) for reporting; "
                        "the production model is still a regression estimator."
                    )
            
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
            st.warning("Model metadata not found. Please run `python3 src/train_model.py` to generate insights.")


if __name__ == "__main__":
    main()
