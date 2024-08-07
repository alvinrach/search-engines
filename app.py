import streamlit as st
from modules import bm25
import json

st.title('Similar Account')

st.warning(
    """
    This app is developed to find similarty between TP standard account name and similar account names using bm25 algorithm.

    The app is purely prototype, still in development cycle, and need to clarify to client whether this is what the client wants or not.
    """
)

with open("assets.json", "r") as read_content: 
    assets = json.load(read_content)

option = st.selectbox(
    "Select TP Standard Account Name",
    assets,
    index=None,
    placeholder="Select one...",
)

st.markdown('# Top 2 Most Similar Items:')

try:
    results = bm25(query_item=option)
    for j,i in enumerate(results):
        st.write(f'{j+1}. {i}')
except:
    st.write('Select the account name first')