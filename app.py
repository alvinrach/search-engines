import streamlit as st
from modules import new_bm25, new_bm25_chroma
import json

st.title('Similar Account Term Prediction💫')

st.info(
    """
    This app is developed to find similarty between TP standard account name and similar account names using bm25 algorithm.

    Initially we create it using only bm25.

    But we will combine with vector database with several algorithm recipe✨ so the prediction can be better.
    """
)

# Initialize session state for results if not already done
if 'bm25_results' not in st.session_state:
    st.session_state.bm25_results = None

if 'bm25_chroma_results' not in st.session_state:
    st.session_state.bm25_chroma_results = None

# BM25 Search
st.markdown('# Using Only bm25😕')
searching = st.text_input("Type Account Name", placeholder='e.g. Net Sales', key='bm25')

if st.button('Search', key='bm25_button'):
    st.session_state.bm25_results = new_bm25(searching)

if st.session_state.bm25_results is not None:
    st.dataframe(st.session_state.bm25_results)

# BM25 & Chroma Search
st.markdown('# Using bm25 & chroma😊')
searching2 = st.text_input("Type Account Name", placeholder='e.g. Net Sales', key='bm25_chroma')

if st.button('Search', key='bm25_chroma_button'):
    st.session_state.bm25_chroma_results = new_bm25_chroma(searching2)

if st.session_state.bm25_chroma_results is not None:
    st.dataframe(st.session_state.bm25_chroma_results)

st.markdown('# Explanation and Hints')
st.success(
    """
    Try to input Net Sales and see the Difference.

    We can see if we use only bm25, another key tag will be shown - Other Income (Expense) - and it became the first search to be shown.

    But we will involve also semantic search, so right now if we have more than 2 products with bm25 score more than 0, it would sort also by the semantic distances.

    The only difference is, for the semantic distances, the lesser the better, for the bm25 score, the more the better.
    """
)