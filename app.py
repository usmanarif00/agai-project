import streamlit as st
from AGAI_Assignment import run_pipeline   # IMPORTANT (we will fix naming if needed)

st.title("🧠 Tech Support Hub (AGAI Project)")

query = st.text_input("Enter your query")

if st.button("Run"):
    if query.strip() == "":
        st.warning("Please enter a query")
    else:
        with st.spinner("Running AI agents..."):
            result = run_pipeline(query)

        st.subheader("Final Answer")
        st.write(result)
