"""Habitat: a considered workspace for property valuation."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.agent import PropertyAdvisorAgent, _parse_inr
from src.pdf_report import build_property_report
from src.validator import PropertyInputValidator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/house_model.pkl"
METADATA_PATH = ROOT / "assets/model_metadata.json"
DEFAULTS = dict(area=3000, bedrooms=3, bathrooms=2, stories=2, parking=1,
                mainroad=True, guestroom=False, basement=False,
                hotwaterheating=False, airconditioning=True)
PRESETS = {
    "Family home": DEFAULTS,
    "Compact home": {**DEFAULTS, "area": 1800, "bedrooms": 2, "bathrooms": 1, "stories": 1},
    "Large residence": {**DEFAULTS, "area": 6000, "bedrooms": 4, "bathrooms": 3, "parking": 2, "guestroom": True},
}
LABELS = {"area": "Floor area", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
          "stories": "Floors", "parking": "Parking", "mainroad": "Main road access",
          "guestroom": "Guest room", "basement": "Basement", "hotwaterheating": "Hot water heating",
          "airconditioning": "Air conditioning"}

st.set_page_config(page_title="Habitat — Property intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")


def html(markup: str) -> None:
    st.html(markup)


def icon(name: str, size: int = 20) -> str:
    paths = {
        "home": '<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1Z"/><path d="M9 21v-8h6v8"/>',
        "spark": '<path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4Z"/>',
        "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
        "check": '<path d="m5 12 4 4L19 6"/>',
        "chart": '<path d="M4 4v16h16M8 15v-4m5 4V7m5 8v-6"/>',
        "shield": '<path d="M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6Z"/><path d="m8 12 3 3 5-6"/>',
    }
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#9bb879" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{paths.get(name, paths["home"])}</svg>'
    encoded = base64.b64encode(svg.encode()).decode()
    return f'<img alt="" aria-hidden="true" width="{size}" height="{size}" src="data:image/svg+xml;base64,{encoded}" />'


def _format_inr(value: float) -> str:
    return f"₹{float(value):,.0f}"


def compact_inr(value: float) -> str:
    return f"₹{value / 10_000_000:.2f} Cr" if value >= 10_000_000 else f"₹{value / 100_000:.2f} L"


@st.cache_resource
def load_advisor_agent():
    return PropertyAdvisorAgent()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def image_uri():
    path = ROOT / "assets/ui/residence.png"
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def apply_preset(name: str = "Family home") -> None:
    for key, value in PRESETS[name].items():
        st.session_state[f"pv_{key}"] = value
    st.session_state.pv_preset = name


def reset_workspace() -> None:
    apply_preset()
    st.session_state.pop("pv_last_result", None)
    st.session_state.pv_sample = True
    st.session_state.pv_page = "Valuation studio"


def section_title(number: str, title: str, subtitle: str = "") -> None:
    html(f'<div class="section-title"><span class="section-number">{number}</span><div><h3>{title}</h3><p>{subtitle}</p></div></div>')


class ValuationApp:
    def __init__(self):
        self.validator = PropertyInputValidator()
        try:
            self.metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        except (OSError, json.JSONDecodeError):
            self.metadata = {}
        try:
            self.model = load_model()
        except Exception as exc:
            self.model = None
            st.error(f"The valuation model could not be loaded: {exc}")
        for key, value in DEFAULTS.items():
            st.session_state[f"pv_{key}"] = st.session_state.get(f"pv_{key}", value)
        st.session_state.pv_preset = st.session_state.get("pv_preset", "Family home")
        st.session_state.setdefault("pv_sample", True)
        if "pv_last_result" not in st.session_state and self.model is not None:
            self._run_valuation(**DEFAULTS)

    def sidebar(self):
        with st.sidebar:
            html(f'<div class="brand"><div class="brand-mark">{icon("home", 25)}</div><div>habitat<span>PROPERTY INTELLIGENCE</span></div></div>')
            html('<div class="workspace-badge"><span class="workspace-initial">P</span><div>Personal workspace<small>Property valuation advisor</small></div></div><div class="nav-label">WORKSPACE</div>')
            page = st.radio("Workspace navigation", ["Valuation studio", "Comparable sales", "Model insights"], key="pv_page", label_visibility="collapsed")
            st.button("＋  New valuation", on_click=reset_workspace, use_container_width=True)
            html(f'<div class="sidebar-note"><span class="note-icon">{icon("spark", 23)}</span><h4>A little more clarity.<br>A smarter next move.</h4><p>Turn property details into an informed perspective.</p><div class="note-line"></div><span class="system-status"><i></i> {"Valuation engine ready" if self.model is not None else "Model unavailable"}</span></div>')
            html('<div class="sidebar-footer"><span class="avatar">PV</span><div>Property workspace<small>India · INR</small></div><span class="footer-dot">•••</span></div>')
        return page

    def header(self, page):
        html(f'<div class="topbar"><div>Workspace <span>/</span> <strong>{page}</strong></div><div class="topbar-right"><span class="local-pill"><i></i> Local knowledge base</span><span class="top-avatar">PV</span></div></div>')

    def hero(self):
        uri = image_uri()
        html(f'''<section class="hero"><div class="hero-copy"><div class="eyebrow"><span></span> A NEW PERSPECTIVE ON PROPERTY</div><h1>Know its value.<br><em>See its potential.</em></h1><p>Thoughtful property decisions start with a clearer picture.<br> Explore valuations, comparable sales, and the story behind the numbers.</p><div class="hero-bottom"><span>{icon("shield", 16)} Model-backed valuations</span><b>Made for your next move {icon("arrow", 17)}</b></div></div><div class="hero-visual" style="background-image:url('{uri}')"><span class="image-tag">THE ART OF LIVING WELL</span><div class="image-caption"><span>Spaces with possibility.</span><small>Illustrative architecture · AI-generated</small></div><span class="image-arrow">↗</span></div></section>''')
        html('<div class="workflow"><div><span class="step active">01</span><strong>Define your property</strong></div><span class="flow-line"></span><div><span class="step">02</span>Discover its value</div><span class="flow-line"></span><div><span class="step">03</span>Make an informed move</div><span class="workflow-end">YOUR NEXT CHAPTER STARTS HERE ↗</span></div>')

    def form(self):
        with st.container(border=True, key="property_card"):
            section_title("01", "The property", "A few details. A more meaningful estimate.")
            st.selectbox("Start with a property profile", list(PRESETS), key="pv_preset", on_change=lambda: apply_preset(st.session_state.pv_preset))
            with st.form("valuation_form", border=False):
                st.number_input("Floor area · sq ft", min_value=300, max_value=25000, step=50, key="pv_area")
                cols = st.columns(2)
                for idx, (field, maximum) in enumerate([("bedrooms", 10), ("bathrooms", 6), ("stories", 5), ("parking", 5)]):
                    with cols[idx % 2]:
                        st.number_input(LABELS[field], min_value=0 if field == "parking" else 1, max_value=maximum, step=1, key=f"pv_{field}")
                html('<div class="amenity-heading">The little extras <span>AMENITIES</span></div>')
                cols = st.columns(2)
                for idx, field in enumerate(["mainroad", "airconditioning", "guestroom", "basement", "hotwaterheating"]):
                    with cols[idx % 2]:
                        st.checkbox(LABELS[field], key=f"pv_{field}")
                submitted = st.form_submit_button("Discover property value  →", type="primary", use_container_width=True, disabled=self.model is None)
                if submitted:
                    st.session_state.pv_sample = False
                    self._run_valuation(**{k: st.session_state[f"pv_{k}"] for k in DEFAULTS})
            html('<div class="form-footnote">Your estimate uses 10 property features. No account needed.</div>')

    def result(self):
        result = st.session_state.get("pv_last_result")
        if not result:
            st.info("Enter your property details to generate a valuation.")
            return
        if result["status"] == "invalid":
            st.error("These details need another look.")
            for message in result["validation"].errors:
                st.error(message)
            return
        if result["status"] == "error":
            st.error("We couldn't complete the valuation. Please try again.")
            st.caption(result.get("error", ""))
            return
        price = result["prediction"]
        mae = self.metadata.get("metrics", {}).get("mae")
        raw = result["raw_inputs"]
        sample = st.session_state.get("pv_sample", False)
        is_stale = any(st.session_state[f"pv_{k}"] != raw[k] for k in DEFAULTS)
        badge = "SAMPLE VALUATION" if sample else "YOUR VALUATION"
        if is_stale:
            st.info("Details changed. Select ‘Discover property value’ to update this estimate.")
        range_html = f'<div class="range-heading"><span>Indicative range · ± model MAE</span><strong>{compact_inr(max(0, price-mae))} – {compact_inr(price+mae)}</strong></div><div class="range-track"><span></span><i></i></div><div class="range-labels"><span>{compact_inr(max(0, price-mae))}</span><span>MODEL ESTIMATE</span><span>{compact_inr(price+mae)}</span></div>' if mae else ''
        html(f'''<div class="valuation-card"><div class="valuation-top"><span>{icon("spark", 18)} THE BIG PICTURE</span><span class="result-badge">{badge}</span></div><div class="estimate-label">Estimated property value</div><div class="estimate-value">{compact_inr(price)}<span>INR</span></div><p class="estimate-summary">{raw['bedrooms']} bedrooms <span>·</span> {raw['bathrooms']} bathrooms <span>·</span> {raw['area']:,} sq ft</p>{range_html}<div class="valuation-disclaimer">Historical model estimate. The range shows average model error,<br>not a confidence interval or a live market quote.</div></div>''')
        html(f'<div class="mini-metrics"><div><span>VALUE PER SQ FT</span><strong>{_format_inr(price/raw["area"])}</strong><small>Based on this estimate</small></div><div><span>COMPARABLE SALES</span><strong>{len(result["comps"]):02d}<b> properties</b></strong><small>Matched from our knowledge base</small></div></div>')
        warnings = result["validation"].warnings
        html(f'<div class="validation-strip">{icon("shield", 18)} <strong>{"Review recommended" if warnings else "Property checks passed"}</strong><span>{str(len(warnings))+" flags to review" if warnings else "No input anomalies"}</span></div>')
        if warnings:
            with st.expander("Review property checks", expanded=True):
                for warning in warnings:
                    st.warning(warning)
        try:
            report = build_property_report(result["inputs"], price, result["advice"], warnings, self.metadata, result["comps"])
            st.download_button("↓  Download investment brief", report, file_name="habitat-investment-brief.pdf", mime="application/pdf", use_container_width=True)
        except Exception:
            st.warning("Your valuation is ready, but the PDF could not be generated. Please try again.")
        html('<div class="export-caption">Your valuation, comparable sales & advisory. In one PDF.</div>')

    def intelligence(self):
        result = st.session_state.get("pv_last_result", {})
        left, right = st.columns([1.2, 1], gap="medium")
        with left, st.container(border=True, key="advisor_card"):
            html(f'<div class="card-heading"><span class="soft-icon">{icon("spark")}</span><h3>Beyond the numbers</h3><span class="small-tag">ADVISOR NOTES</span></div>')
            if result.get("status") == "ok":
                if result.get("advisory_error"):
                    st.info("The valuation is ready. Advisory context is temporarily unavailable.")
                else:
                    text = result["advice"].split("Recommendation: ")[-1] if "Recommendation: " in result["advice"] else result["advice"]
                    st.markdown(text)
                    with st.expander("Explore the full property narrative"):
                        st.markdown(result["advice"])
                st.caption("Rule-based guidance · Static knowledge base · Verify local pricing")
            else:
                st.caption("Complete a valuation to explore the property narrative.")
        with right, st.container(border=True, key="drivers_card"):
            html(f'<div class="card-heading"><span class="soft-icon">{icon("chart")}</span><h3>What moves the value</h3></div>')
            html('<p class="driver-intro">The strongest influences across the trained model.</p>')
            importance = self.metadata.get("feature_importance", {})
            for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]:
                html(f'<div class="driver"><span>{LABELS.get(key, key)}</span><div><i style="width:{value/max(importance.values())*100:.1f}%"></i></div><b>{value*100:.1f}%</b></div>')
            html('<p class="driver-note">Global feature importance, not an individual price breakdown.</p>')

    def comparables(self):
        section_title("02", "A little market perspective", "Similar properties from the project's historical knowledge base.")
        result = st.session_state.get("pv_last_result", {})
        comps = result.get("comps") or []
        if not comps:
            st.info("Complete a valuation in the studio to find comparable properties.")
            return
        html('<div class="context-note">HISTORICAL CONTEXT <span>These records are static examples, not live listings or verified current transactions.</span></div>')
        cols = st.columns(len(comps))
        for i, (col, comp) in enumerate(zip(cols, comps)):
            with col, st.container(border=True):
                html(f'<div class="comp-icon">{icon("home", 28)}<span>COMPARABLE 0{i+1}</span></div>')
                st.subheader(str(comp.get("location", "Unknown location")))
                price = _parse_inr(comp.get("price"))
                st.metric("Recorded sale price", compact_inr(price) if price is not None else "Unavailable")
                st.write(f"{comp.get('area', '—'):,} sq ft · {comp.get('bedrooms', '—')} bed · {comp.get('bathrooms', '—')} bath")
                st.caption(f"Recorded: {comp.get('date', 'Unknown')}")
        st.dataframe(pd.DataFrame(comps), hide_index=True, use_container_width=True)

    def insights(self):
        section_title("03", "A clearer view of the model", "Understand the evidence behind your estimate.")
        metrics = self.metadata.get("metrics")
        if not metrics:
            st.info("Model performance metadata is unavailable.")
            return
        columns = st.columns(4)
        for col, label, value in zip(columns, ["R² score", "Mean absolute error", "Root mean squared error", "Within 10% of actual"], [f"{metrics['r2']:.3f}", compact_inr(metrics["mae"]), compact_inr(metrics["rmse"]), f"{metrics.get('within_10_pct', 0)*100:.1f}%"]):
            with col, st.container(border=True):
                st.metric(label, value)
        left, right = st.columns([1.2, 1], gap="large")
        with left, st.container(border=True):
            st.subheader("What the model pays attention to")
            importance = self.metadata.get("feature_importance", {})
            for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                html(f'<div class="driver"><span>{LABELS.get(key, key)}</span><div><i style="width:{value/max(importance.values())*100:.1f}%"></i></div><b>{value*100:.1f}%</b></div>')
        with right:
            st.subheader("Built on patterns. Grounded in context.")
            st.write("The Random Forest model combines decision trees to estimate a price from ten physical and amenity features. The advisory layer retrieves relevant text and comparable sales from a local knowledge base, then applies explicit decision rules.")
            st.write("Feature importance describes the model as a whole. It does not explain an individual property's price contribution. The model does not use a property's address or live market data.")
            st.caption("The model's average absolute error is shown alongside every valuation. Use estimates as directional starting points.")
            cm = self.metadata.get("classification_metrics", {})
            if cm:
                with st.expander("Additional evaluation · High-value classification"):
                    st.write("These metrics use a derived price-above-median label; production predictions remain regression estimates.")
                    st.json(cm)

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



def main():
    html(f'<style>{(ROOT / "assets/ui/styles.css").read_text()}</style>')
    app = ValuationApp()
    page = app.sidebar()
    app.header(page)
    if page == "Valuation studio":
        app.hero()
        left, right = st.columns([1, 1.06], gap="large")
        with left:
            app.form()
        with right:
            app.result()
        app.intelligence()
    elif page == "Comparable sales":
        app.comparables()
    else:
        app.insights()
    html('<footer class="page-footer"><span>habitat <b> / </b> A considered approach to property.</span><span>PROPERTY VALUATION AGENTIC ADVISOR</span></footer>')


if __name__ == "__main__":
    main()
