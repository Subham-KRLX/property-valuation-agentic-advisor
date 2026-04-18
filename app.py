"""
Streamlit Cloud default entrypoint.

The project source lives in `src/`. Keeping this thin wrapper allows the
Streamlit Cloud configuration to continue pointing at `app.py`.
"""

from src.app import main


main()

