"""Exercise the native Streamlit workflow against the bundled model."""
from streamlit.testing.v1 import AppTest


def app():
    return AppTest.from_file("app.py", default_timeout=60).run()


def test_profile_submission_and_navigation_preserve_valuation():
    at = app()
    assert not at.exception
    initial = at.session_state["pv_last_result"]["prediction"]
    assert initial > 0
    assert at.session_state["pv_sample"]
    at.selectbox(key="pv_preset").select("Large residence").run()
    assert at.number_input(key="pv_area").value == 6000
    at.button(key="FormSubmitter:valuation_form-Discover property value  →").click().run()
    result = at.session_state["pv_last_result"]
    assert result["status"] == "ok"
    assert result["prediction"] != initial
    assert result["raw_inputs"]["area"] == 6000
    assert not at.session_state["pv_sample"]
    at.radio(key="pv_page").set_value("Comparable sales").run()
    assert not at.exception
    assert len(at.dataframe) == 1
    at.radio(key="pv_page").set_value("Model insights").run()
    assert not at.exception
    assert len(at.metric) == 4
    at.radio(key="pv_page").set_value("Valuation studio").run()
    assert at.number_input(key="pv_area").value == 6000
    assert at.session_state["pv_last_result"]["prediction"] == result["prediction"]


def test_invalid_input_then_reset_recovers():
    at = app()
    at.number_input(key="pv_area").set_value(300)
    at.number_input(key="pv_bedrooms").set_value(10)
    at.button(key="FormSubmitter:valuation_form-Discover property value  →").click().run()
    assert at.session_state["pv_last_result"]["status"] == "invalid"
    assert len(at.error) > 0
    assert not at.exception
    at.radio(key="pv_page").set_value("Model insights").run()
    next(b for b in at.button if "New valuation" in b.label).click().run()
    assert at.radio(key="pv_page").value == "Valuation studio"
    assert at.number_input(key="pv_area").value == 3000
    assert at.session_state["pv_last_result"]["status"] == "ok"
    assert at.session_state["pv_sample"]
