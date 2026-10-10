"""Presentation helpers for the valuation workspace."""

from html import escape
from statistics import median

import pandas as pd
import streamlit as st

from src.agent import _parse_inr


FEATURE_LABELS = {
    "area": "Floor area", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
    "stories": "Floors", "parking": "Parking", "mainroad": "Road access",
    "guestroom": "Guest room", "basement": "Basement",
    "hotwaterheating": "Hot water", "airconditioning": "Air conditioning",
}

PRESETS = {
    "Compact home": dict(area=1800, bedrooms=2, bathrooms=1, stories=1, parking=1,
                         mainroad=True, guestroom=False, basement=False,
                         hotwaterheating=False, airconditioning=False),
    "Family home": dict(area=3000, bedrooms=3, bathrooms=2, stories=2, parking=2,
                        mainroad=True, guestroom=True, basement=False,
                        hotwaterheating=False, airconditioning=True),
    "Large residence": dict(area=6000, bedrooms=4, bathrooms=3, stories=2, parking=3,
                            mainroad=True, guestroom=True, basement=True,
                            hotwaterheating=True, airconditioning=True),
}
PRESETS["Happy path"] = dict(PRESETS["Family home"], area=2500, guestroom=False)
PRESETS["Outlier demo"] = dict(PRESETS["Large residence"], area=5000, bedrooms=1,
                              bathrooms=1, parking=1, mainroad=False,
                              guestroom=False, hotwaterheating=False)


def format_inr(value):
    if value is None:
        return "Unavailable"
    return f"₹{float(value):,.0f}"


def styles():
    st.markdown("""
    <style>
    .stMainBlockContainer {max-width: 1280px; padding-top: 4rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #dce5e1;}
    .workspace-header {border-bottom: 1px solid #dce5e1; padding-bottom: 1.4rem; margin-bottom: 1.5rem;}
    .workspace-header h1 {font-size: 2rem; line-height: 1.25; margin: .35rem 0; letter-spacing: 0;}
    .workspace-header p {color: #596b65; margin: 0;}
    .eyebrow {color: #087f68; font-size: .75rem; font-weight: 700; letter-spacing: 0; text-transform: uppercase;}
    .estimate {background: #edf7f2; border-left: 4px solid #087f68; padding: 1.5rem; margin: 1rem 0;}
    .estimate-value {font-size: 2.6rem; font-weight: 700; line-height: 1.3; overflow-wrap: anywhere;}
    .estimate p {margin: .4rem 0 0; color: #52665e;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    [data-testid="stTabs"] {margin-top: 1.5rem;}
    .pv-splash {position: fixed; inset: 0; z-index: 9999; display: grid;
      place-items: center; pointer-events: none; background: #f3f6f5;
      animation: welcome-out .5s .4s forwards;}
    .pv-splash-card {padding: 1.5rem; border: 1px solid #dce5e1; background: white; border-radius: 8px;}
    .pv-splash-title {font-weight: 700; color: #087f68;}
    .pv-splash-sub {color: #596b65; margin-top: .4rem;}
    @keyframes welcome-out {to {opacity: 0; visibility: hidden;}}
    @media(prefers-reduced-motion: reduce) {.pv-splash {display: none;}}
    @media(max-width: 640px) {
      .stMainBlockContainer {padding: 4rem 1rem 2rem;}
      .workspace-header h1 {font-size: 1.6rem;}
      .estimate-value {font-size: 2rem;}
      .estimate {padding: 1rem;}
    }
    </style>
    """, unsafe_allow_html=True)


def header():
    st.markdown("""
    <header class="workspace-header">
      <div class="eyebrow">Property intelligence / Valuation workspace</div>
      <h1>Property Valuation Advisor</h1>
      <p>A clearer view of property value, comparable evidence, and model performance.</p>
    </header>
    """, unsafe_allow_html=True)


def apply_preset():
    for feature, value in PRESETS[st.session_state.pv_preset].items():
        st.session_state[f"pv_{feature}"] = value
    st.session_state.pv_last_result = None


def reset_workspace():
    for key in list(st.session_state):
        if key.startswith("pv_"):
            del st.session_state[key]


def empty_state():
    st.subheader("Your next property analysis")
    st.caption("Historical housing model · Local comparable-sales knowledge base")
    st.markdown("Choose a property profile or enter the details to begin.")
    cols = st.columns(3)
    for col, (name, details) in zip(cols, PRESETS.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.metric("Floor area", f"{details['area']:,} sq ft")
                st.caption(f"{details['bedrooms']} bedrooms · {details['bathrooms']} bathrooms")
    st.divider()
    st.subheader("Evidence behind the estimate")
    left, right = st.columns(2)
    with left:
        st.markdown("**Property valuation**")
        st.write("A trained Random Forest evaluates physical attributes and amenities.")
    with right:
        st.markdown("**Comparable context**")
        st.write("Local knowledge-base records ground the rule-based investment summary.")
    st.caption("Research prototype. Estimates use historical data; comparable records are not a live market feed.")


def result_overview(result, metadata=None):
    prediction = result["prediction"]
    raw = result["raw_inputs"]
    st.markdown(f"""
    <section class="estimate">
      <div class="eyebrow">Estimated property value</div>
      <div class="estimate-value">{escape(format_inr(prediction))}</div>
      <p>{int(raw['area']):,} sq ft · {int(raw['bedrooms'])} bedrooms · {int(raw['bathrooms'])} bathrooms</p>
    </section>
    """, unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Value per sq ft", format_inr(prediction / raw["area"]))
    b.metric("Parking spaces", str(raw["parking"]))
    c.metric("Amenities included", f"{result['amenities']} / 5")
    st.caption(f"Analysis generated {result['generated_at']:%d %b %Y, %H:%M}. Historical model estimate, not an appraisal.")
    mae = (metadata or {}).get("metrics", {}).get("mae")
    if isinstance(mae, (int, float)):
        with st.expander("Historical model error"):
            st.metric("Estimate ± average test error", f"{format_inr(max(0, prediction - mae))} – {format_inr(prediction + mae)}")
            st.caption("Illustrative sensitivity band using held-out MAE. This is not a calibrated confidence interval or a guaranteed price range.")
    warnings = result["validation"].warnings
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("Input checks passed. No input anomalies were flagged.")


def advisory(text):
    summary, marker, remainder = text.partition("Retrieved market context (knowledge base):")
    if not marker:
        st.write(text)
        return
    context, recommendation_marker, recommendation = remainder.rpartition("Recommendation:")
    st.write(summary)
    if recommendation_marker:
        st.write(f"Recommendation: {recommendation}")
    else:
        context = remainder
    with st.expander("Retrieved source context"):
        st.markdown(context)


def comparables(result):
    st.subheader("Comparable sales")
    st.caption("Matched records from the local knowledge base. Locations and sale dates may differ from the subject property.")
    comps = result.get("comps") or []
    if not comps:
        st.info("No comparable sales available for this property.")
        return
    frame = pd.DataFrame(comps)
    prices = pd.to_numeric(frame.get("price", pd.Series(dtype=float)).map(_parse_inr), errors="coerce")
    valid = prices.dropna()
    if not valid.empty:
        a, b = st.columns(2)
        comp_median = median(valid)
        a.metric("Median comparable price", format_inr(comp_median))
        b.metric("Estimate vs comparable median", f"{(result['prediction'] / comp_median - 1) * 100:+.1f}%")
        chart = pd.DataFrame({"Property": [str(c.get("location", "Comparable")) for c in comps],
                              "Price (INR)": prices})
        chart = pd.concat([chart, pd.DataFrame({"Property": ["Subject estimate"], "Price (INR)": [result["prediction"]]})], ignore_index=True)
        st.bar_chart(chart, x="Property", y="Price (INR)", color="#087F68")
    if "price" in frame:
        frame["price"] = prices.map(lambda x: format_inr(x) if pd.notna(x) else "Unavailable")
    columns = [name for name in ("location", "date", "price", "area", "bedrooms", "bathrooms") if name in frame]
    frame = frame[columns].rename(columns={"location": "Location", "price": "Sale price", "date": "Sale date", "area": "Area (sq ft)", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms"})
    st.dataframe(frame, width="stretch", hide_index=True)


def model_insights(metadata):
    st.subheader("Model evidence")
    if not metadata:
        st.info("Model evaluation metadata is unavailable.")
        return
    metrics = metadata["metrics"]
    a, b, c = st.columns(3)
    a.metric("Test R²", f"{metrics['r2']:.3f}")
    b.metric("Mean absolute error", format_inr(metrics["mae"]))
    c.metric("Within 20% of actual", f"{metrics.get('within_20_pct', 0) * 100:.1f}%")
    st.caption("Held-out evaluation metrics describe past test performance, not confidence for an individual estimate.")
    st.subheader("What the model weighs")
    importance = metadata.get("feature_importance", {})
    frame = pd.DataFrame({"Feature": [FEATURE_LABELS.get(k, k) for k in importance], "Importance": list(importance.values())})
    st.bar_chart(frame.sort_values("Importance", ascending=False), x="Feature", y="Importance", horizontal=True, color="#087F68")
    st.caption("Global feature importance is not a causal explanation of this property's price.")
    candidates = metadata.get("benchmark_summary", {}).get("candidates", {})
    if candidates:
        st.subheader("Benchmark comparison")
        rows = [{"Model": item["label"], "R²": item["metrics"]["r2"],
                 "MAE (INR)": item["metrics"]["mae"], "RMSE (INR)": item["metrics"]["rmse"]} for item in candidates.values()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    training = metadata.get("training", {})
    with st.expander("Training and methodology"):
        st.write("Random Forest regression with deterministic TF-IDF retrieval and a template-based advisor. No language model or API key is required.")
        st.write(f"Training records: {training.get('train_row_count', 'Unavailable')} · Test records: {training.get('test_row_count', 'Unavailable')}")
        st.write("The housing model has no location feature. Bangalore-focused knowledge-base context does not make the prediction a Bangalore-specific appraisal.")
