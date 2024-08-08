import streamlit as st
from modules import new_bm25, new_bm25_chroma
import json

st.title('Similar Account')

st.warning(
    """
    This app is developed to find similarty between TP standard account name and similar account names using bm25 algorithm.

    The app is purely prototype, still in development cycle, and need to clarify to client whether this is what the client wants or not.
    """
)

# Initialize session state for results if not already done
if 'bm25_results' not in st.session_state:
    st.session_state.bm25_results = None

if 'bm25_chroma_results' not in st.session_state:
    st.session_state.bm25_chroma_results = None

# BM25 Search
st.markdown('# Using Only bm25')
searching = st.text_input("Type Account Name", placeholder='e.g. Turnover', key='bm25')

if st.button('Search', key='bm25_button'):
    st.session_state.bm25_results = new_bm25(searching)

if st.session_state.bm25_results is not None:
    st.dataframe(st.session_state.bm25_results)

# BM25 & Chroma Search
st.markdown('# Using bm25 & chroma')
searching2 = st.text_input("Type Account Name", placeholder='e.g. Turnover', key='bm25_chroma')

if st.button('Search', key='bm25_chroma_button'):
    st.session_state.bm25_chroma_results = new_bm25_chroma(searching2)

if st.session_state.bm25_chroma_results is not None:
    st.dataframe(st.session_state.bm25_chroma_results)