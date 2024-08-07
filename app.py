import streamlit as st
from modules import new_bm25
import json

st.title('Similar Account')

st.warning(
    """
    This app is developed to find similarty between TP standard account name and similar account names using bm25 algorithm.

    The app is purely prototype, still in development cycle, and need to clarify to client whether this is what the client wants or not.
    """
)

searching = st.text_input("Type Account Name", placeholder='e.g. Turnover')

if st.button('Search'):
    st.dataframe(new_bm25(searching))