from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import json

# Ensure you have the necessary NLTK data
nltk.download('punkt')

def bm25(query_item):
    # Your list of items
    with open("assets.json", "r") as read_content: 
        assets = json.load(read_content)

    # import json
    # with open("assets.json", "w") as final:
    #     json.dump(assets, final)
    
    # Tokenize the items
    tokenized_assets = [word_tokenize(item.lower()) for item in assets]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_assets)

    # Function to find the top n most similar items
    def find_top_n_similar(item, bm25, items, n=5):
        tokenized_query = word_tokenize(item.lower())
        scores = bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[-n:][::-1]  # Get indices of top n scores
        top_n_items = [items[i] for i in top_n_indices]
        return top_n_items

    # Test the function with an item from the list
    top_n = 2  # Number of most similar items to find
    most_similar_items = find_top_n_similar(query_item, bm25, assets, top_n)
    print(f"The top {top_n} most similar items to '{query_item}' are:")
    for item in most_similar_items:
        print(item)
    
    return most_similar_items