from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import json
import pandas as pd
import chromadb
from chromadb.config import Settings

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

def new_bm25(query):
    file_path = 'fin_template_2.csv'
    data = pd.read_csv(file_path)
    # data.labels = data.labels.fillna(data.account_name)
    data = data.dropna()
    data = data.reset_index(drop=True)
    data.labels = data.labels.str.lower()

    data['labels'] = data['labels'].apply(lambda x: x.split(', '))
    bm25 = BM25Okapi(data['labels'])

    def search_labels(query):
        query = query.lower()
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        data['bm25_scores'] = scores 
        top_n = np.argsort(scores)[::-1]  # Get indices of top scores in descending order

        results = data.loc[top_n, ['account_name', 'account_code', 'bm25_scores']].head(10)  # Return top 10 results
        if len(results[results.bm25_scores>0])>0:
            results = results[results.bm25_scores>0]
        else:
            results = data.loc[top_n, ['account_name', 'account_code', 'bm25_scores']].head(1)  # Return top 10 results

        return results

    # query = "goodwill"
    search_results = search_labels(query)
    return search_results

def new_bm25_chroma(query):
    file_path = 'fin_template_2.csv'
    data = pd.read_csv(file_path)
    data = data.dropna()
    data = data.reset_index(drop=True)
    data.labels = data.labels.str.lower()

    data['labels'] = data['labels'].apply(lambda x: x.split(', '))
    bm25 = BM25Okapi(data['labels'])

    # Initialize Chroma DB
    chroma_client = chromadb.Client(Settings())
    collection = chroma_client.get_or_create_collection(name="financial_data")

    # Add documents to Chroma DB if not already added
    if len(collection.get()['documents']) == 0:
        documents = data['labels'].apply(lambda x: ' '.join(x)).tolist()
        document_ids = [str(i) for i in range(len(documents))]
        collection.add(documents=documents, ids=document_ids)

    def get_chroma_distances(query):
        query = query.lower()
        result = collection.query(query_texts=[query], n_results=len(data))
        distances = [0] * len(data)
        for idx, distance in zip(result['ids'][0], result['distances'][0]):
            distances[int(idx)] = distance  # Convert distance to score
        return distances

    def search_labels(query):
        query = query.lower()
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        chroma_distances = get_chroma_distances(query)
        data['bm25_scores'] = scores
        data['chroma_distances'] = chroma_distances

        top_n = np.argsort(scores)[::-1]  # Get indices of top scores in descending order
        results = data.loc[top_n, ['account_name', 'account_code', 'bm25_scores', 'chroma_distances']]

        number_bm25_relevant = len(results[results.bm25_scores > 0])
        if number_bm25_relevant == 0 :
            top_n = np.argsort(chroma_distances)#[::-1]
            results = data.loc[top_n, ['account_name', 'account_code', 'bm25_scores', 'chroma_distances']].head(1)
        elif number_bm25_relevant == 1:
            results = results[results.bm25_scores > 0]
        else:
            results = data.loc[top_n, ['account_name', 'account_code', 'bm25_scores', 'chroma_distances']].head(number_bm25_relevant)
            results = results.sort_values('chroma_distances', ascending=True).head(1)

        return results

    search_results = search_labels(query)
    return search_results
