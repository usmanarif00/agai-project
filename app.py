import streamlit as st

st.title("🧠 Tech Support Hub (AGAI Project)")

st.write("AGAI Assignment backend runs in notebook.")

query = st.text_input("Enter your query")

if st.button("Run"):
    st.success("Notebook contains full LangGraph system.")
    st.info("Run AGAI Assignment.ipynb to get real results.")
