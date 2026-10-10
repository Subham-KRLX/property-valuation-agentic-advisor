import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
import textwrap
from src.validator import PropertyInputValidator

from src.agent import PropertyAdvisorAgent
from src.pdf_report import build_property_report
from src import dashboard

PAGE_TITLE = "Property Valuation Advisor"
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
    initial_sidebar_state="auto",
)


def _inject_global_styles() -> None:
    dashboard.styles()


def _format_inr(value: float) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return f"₹{value}"


@st.cache_resource
def load_advisor_agent():
    return PropertyAdvisorAgent()


def _render_splash_screen() -> None:
    """
    Displays a lightweight splash overlay once per browser session.

    Notes:
    - Streamlit reruns the script often; we guard with session_state.
    - The overlay self-dismisses via CSS animation (no JS).
    """
    if st.session_state.get("pv_splash_shown"):
        return
    st.session_state.pv_splash_shown = True

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="pv-splash" role="presentation">
              <div class="pv-splash-card">
                <div class="pv-splash-top">
                  <div class="pv-splash-mark">{PAGE_ICON}</div>
                  <div>
                    <div class="pv-splash-title">{PAGE_TITLE}</div>
                    <div class="pv-splash-sub">Valuation • Comps • PDF brief</div>
                  </div>
                </div>
                <div class="pv-splash-bar"><div></div></div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


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
        dashboard.header()

    def render_sidebar(self):
        st.sidebar.markdown("## Property profile")
        st.sidebar.caption("Valuation workspace")
        st.sidebar.selectbox("Sample property", list(dashboard.PRESETS), index=None,
                             placeholder="Choose a profile", key="pv_preset",
                             on_change=dashboard.apply_preset)

        if self.model is None:
            st.sidebar.warning("Model missing — run `python3 src/train_model.py` first.")
            return
        if "pv_last_result" not in st.session_state:
            st.session_state.pv_last_result = None

        with st.sidebar.form("valuation_form"):
            st.markdown("### Property details")
            area = st.number_input(
                "Area (sq ft)",
                min_value=300,
                max_value=25000,
                value=3000,
                key="pv_area",
                step=50,
                help="Built-up area in square feet.",
            )
            col1, col2 = st.columns(2)
            with col1:
                bedrooms = st.number_input(
                    "Bedrooms",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="pv_bedrooms",
                    step=1,
                )
                bathrooms = st.number_input(
                    "Bathrooms",
                    min_value=1,
                    max_value=6,
                    value=2,
                    key="pv_bathrooms",
                    step=1,
                )
            with col2:
                stories = st.number_input(
                    "Stories",
                    min_value=1,
                    max_value=5,
                    value=2,
                    key="pv_stories",
                    step=1,
                )
                parking = st.number_input(
                    "Parking spots",
                    min_value=0,
                    max_value=5,
                    value=1,
                    key="pv_parking",
                    step=1,
                )

            st.markdown("### Amenities")
            mainroad = st.checkbox("Main road access", value=True, key="pv_mainroad")
            guestroom = st.checkbox("Guest room", value=False, key="pv_guestroom")
            basement = st.checkbox("Basement", value=False, key="pv_basement")
            hotwaterheating = st.checkbox("Hot water heating", value=False, key="pv_hotwaterheating")
            airconditioning = st.checkbox("Air conditioning", value=True, key="pv_airconditioning")

            submitted = st.form_submit_button("Estimate value", width="stretch", type="primary")

        st.sidebar.button("Reset property", width="stretch", on_click=dashboard.reset_workspace)

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

    def render_main(self):
        if self.model is None:
            return
        last = st.session_state.get("pv_last_result")
        if not last:
            dashboard.empty_state()
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

        advice = result["advice"]

        dashboard.result_overview(result, self.metadata)
        summary, evidence, export = st.tabs(["Investment summary", "Comparable evidence", "Export report"])
        with summary:
            st.subheader("Investment perspective")
            st.caption("Rule-based summary grounded in retrieved records.")
            dashboard.advisory(advice)
            if result.get("advisory_error"):
                st.warning("The advisory service is unavailable. The model estimate is still available.")
        with evidence:
            dashboard.comparables(result)
        with export:
            self._render_report(result)

    def _render_report(self, result):
        raw_inputs = result["raw_inputs"]
        prediction = result["prediction"]
        advice = result["advice"]
        validation = result["validation"]
        comps = result.get("comps") or []
        st.subheader("Property investment brief")
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
            width="stretch",
        )

def main():
    _inject_global_styles()
    _render_splash_screen()
    app = ValuationApp()
    app.render_sidebar()
    app.render_header()
    valuation, model = st.tabs(["Valuation", "Model evidence"])
    with valuation:
        app.render_main()
    with model:
        dashboard.model_insights(app.metadata)


if __name__ == "__main__":
    main()
