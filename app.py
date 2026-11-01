import streamlit as st

st.set_page_config(page_title="Fraud Detection Studio", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

pages = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("app_pages/performance.py", title="Model performance", icon=":material/monitoring:"),
    st.Page("app_pages/simulator.py", title="Threshold & cost simulator", icon=":material/tune:"),
    st.Page("app_pages/scoring.py", title="Score transactions", icon=":material/policy:"),
    st.Page("app_pages/insights.py", title="Feature insights", icon=":material/insights:"),
    st.Page("app_pages/about.py", title="Method & limits", icon=":material/info:"),
]
st.navigation(pages).run()
