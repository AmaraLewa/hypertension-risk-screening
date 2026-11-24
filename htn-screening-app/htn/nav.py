"""Page registry populated by app.py so views can link to each other."""
PAGES: dict = {}


def go(key: str) -> None:
    import streamlit as st

    page = PAGES.get(key)
    if page is not None:
        st.switch_page(page)
