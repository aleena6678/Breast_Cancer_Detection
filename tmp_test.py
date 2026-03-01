import streamlit as st
import time

st.markdown("""
<style>
/* Add padding to the form to make room for absolute alert */
[data-testid="stForm"] {
    position: relative !important;
    padding-bottom: 70px !important; 
}
/* Position alert absolute so it doesn't push elements down */
[data-testid="stForm"] [data-testid="stAlert"] {
    position: absolute !important;
    bottom: 10px !important;
    left: 20px !important;
    right: 20px !important;
    width: auto !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

with st.form("my_form"):
    st.text_input("Username")
    st.text_input("Password")
    submit = st.form_submit_button("Submit")
    if submit:
        st.error("This is an error message. It should not resize the form.")
