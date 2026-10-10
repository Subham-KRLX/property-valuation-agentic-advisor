"""Exercise the real Streamlit workflow, including state across reruns."""

from streamlit.testing.v1 import AppTest


def test_preset_valuation_and_reset():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    app.selectbox[0].select("Large residence").run()
    assert app.number_input[0].value == 6000
    assert app.number_input[1].value == 4
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["pv_last_result"]
    assert result["status"] == "ok"
    assert result["prediction"] > 0
    assert result["inputs"]["area"] == 6000
    assert result["comps"]
    assert any(metric.label == "Median comparable price" and metric.value != "Unavailable" for metric in app.metric)
    assert app.get("arrow_vega_lite_chart")
    assert app.get("download_button")
    app.button[1].click().run()
    assert not app.exception
    assert app.number_input[0].value == 3000
    assert app.session_state["pv_last_result"] is None


def test_outlier_warning_does_not_hide_valuation():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.number_input[0].set_value(5000)
    app.number_input[1].set_value(1)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["pv_last_result"]["status"] == "ok"
    assert app.warning
    assert any(metric.label == "Value per sq ft" for metric in app.metric)


def test_changing_profile_clears_previous_result():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.button[0].click().run()
    assert app.session_state["pv_last_result"]["status"] == "ok"
    app.selectbox[0].select("Compact home").run()
    assert not app.exception
    assert app.session_state["pv_last_result"] is None
    assert app.number_input[0].value == 1800
