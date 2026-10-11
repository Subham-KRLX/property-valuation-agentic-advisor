# Habitat UI redesign

The interface is implemented in `src/app.py`; visual tokens and responsive styles live in `assets/ui/styles.css`. Both existing Streamlit entrypoints remain available. The prediction feature order, trained model, validation, deterministic advisory rules, and PDF generator are retained.

## Interaction design

- First load performs a real prediction for a 3,000 sq ft sample property, explicitly labeled **Sample valuation**.
- Profiles populate actual widget state. Form submission generates a new estimate. Profile changes show a stale-result notice until submitted.
- New valuation restores the sample defaults and returns to the studio from any navigation view.
- Submitted inputs and results survive navigation between the studio, comparable sales, and model insights.
- The valuation card distinguishes the model's mean absolute error from a statistical confidence interval. Comparable records are labeled as historical/static, and advisory notes identify their rule-based source.
- The download contains the current result and its corresponding inputs, warnings, comps, and metadata.
- Native Streamlit controls retain labels and keyboard interaction. The layout stacks on smaller screens and respects reduced-motion preferences.

## Reproducibility

`scikit-learn` is pinned to 1.8.0 to match the version embedded in the repository's serialized model. Streamlit's minimum version is raised to 1.41 for the modern container and rendering APIs used by this interface.

## Architectural asset

Generated using the built-in Imagegen tool. Saved as `assets/ui/residence.png`. It is decorative, AI-generated architecture, labeled in the interface; it does not depict a comparable listing.

Generation prompt:

> Use case: photorealistic-natural. Asset type: wide editorial architectural photograph for the hero of a premium Indian property valuation dashboard. A striking contemporary luxury residence in Bangalore: warm ivory travertine, sculptural stacked rectilinear volumes, floor to ceiling dark glass, slim vertical timber slats, a mature tropical tree and lush restrained landscaping, warm interior glow. Perspective from low front three-quarter angle, house dominates center and right, natural late-afternoon sunlight, pale sage sky. Quiet luxury, authentic architectural magazine photography, subtle film grain, rich dark green and warm cream palette. Landscape 1536x1024. No text, no logos, no watermark, no charts. This is an illustrative residence, not a real listing.

Fonts: DM Sans and Manrope via Google Fonts, with system sans-serif fallbacks. The architecture is served from the repository, so it does not depend on an external image host.

## Verification

- `python -m pytest tests/ -q`: 20 tests, including profile submission, navigation persistence, invalid inputs, and recovery through reset.
- Browser flow: real property input → model result → comparable records → downloaded PDF.
- Desktop screenshot and mobile screenshot are captured from the running application, not generated mockups.

This change is local to branch `codex/ui-redesign`. It does not update the existing Streamlit Cloud deployment.
