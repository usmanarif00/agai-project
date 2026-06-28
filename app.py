import streamlit as st

st.title("🧠 Tech Support Hub")

query = st.text_input("Enter your query")

if st.button("Run"):
    st.write("Received query:")
    st.write(query)
    st.success("App is working on Streamlit Cloud ✅")
