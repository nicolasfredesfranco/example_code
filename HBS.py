#this code is a script version of HBS_retrieval_assignment.ipynb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#define functions for product search using Tf-IDF
def calculate_tfidf(dataframe):
    """
    Calculate the TF-IDF for combined product name and description.

    Parameters:
    dataframe (pd.DataFrame): DataFrame with product_id, and other product information.

    Returns:
    TfidfVectorizer, csr_matrix: TF-IDF vectorizer and TF-IDF matrix.
    """
    # Combine product name and description to vectorize
    # NOTE: Please feel free to use any combination of columns available, some columns may contain NULL values
    combined_text = dataframe['product_name'] + ' ' + dataframe['product_description']
    vectorizer = TfidfVectorizer()
    # convert combined_text to list of unicode strings
    tfidf_matrix = vectorizer.fit_transform(combined_text.values.astype('U'))
    return vectorizer, tfidf_matrix

def get_top_products(vectorizer, tfidf_matrix, query, top_n=10):
    """
    Get top N products for a given query based on TF-IDF similarity.

    Parameters:
    vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
    tfidf_matrix (csr_matrix): TF-IDF matrix for the products.
    query (str): Search query.
    top_n (int): Number of top products to return.

    Returns:
    list: List of top N product IDs.
    """
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_product_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return top_product_indices

#define functions for evaluating retrieval performance
def map_at_k(true_ids, predicted_ids, k=10):
    """
    Calculate the Mean Average Precision at K (MAP@K).

    Parameters:
    true_ids (list): List of relevant product IDs.
    predicted_ids (list): List of predicted product IDs.
    k (int): Number of top elements to consider.
             NOTE: IF you wish to change top k, please provide a justification for choosing the new value

    Returns:
    float: MAP@K score.
    """
    #if either list is empty, return 0
    if not len(true_ids) or not len(predicted_ids):
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, p_id in enumerate(predicted_ids[:k]):
        if p_id in true_ids and p_id not in predicted_ids[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(true_ids), k)

# get search queries
query_df = pd.read_csv("WANDS/dataset/query.csv", sep='\t')

product_df = pd.read_csv("WANDS/dataset/product.csv", sep='\t')

# get manually labeled groundtruth lables
label_df = pd.read_csv("WANDS/dataset/label.csv", sep='\t')

#group the labels for each query to use when identifying exact matches
grouped_label_df = label_df.groupby('query_id')

# Calculate TF-IDF
vectorizer, tfidf_matrix = calculate_tfidf(product_df)

#Sanity check code block to see if the search results are relevant
#implementing a function to retrieve top K product IDs for a query
def get_top_product_ids_for_query(query):
    top_product_indices = get_top_products(vectorizer, tfidf_matrix, query, top_n=10)
    top_product_ids = product_df.iloc[top_product_indices]['product_id'].tolist()
    return top_product_ids

#define the test query
query = "armchair"

#obtain top product IDs
top_product_ids = get_top_product_ids_for_query(query)

print(f"Top products for '{query}':")
for product_id in top_product_ids:
    product = product_df.loc[product_df['product_id'] == product_id]
    print(product_id, product['product_name'].values[0])

#implementing a function to retrieve exact match product IDs for a query_id
def get_exact_matches_for_query(query_id):
    query_group = grouped_label_df.get_group(query_id)
    exact_matches = query_group.loc[query_group['label'] == 'Exact']['product_id'].values
    return exact_matches

#applying the function to obtain top product IDs and adding top K product IDs to the dataframe 
query_df['top_product_ids'] = query_df['query'].apply(get_top_product_ids_for_query)

#adding the list of exact match product_IDs from labels_df
query_df['relevant_ids'] = query_df['query_id'].apply(get_exact_matches_for_query)

#now assign the map@k score
query_df['map@k'] = query_df.apply(lambda x: map_at_k(x['relevant_ids'], x['top_product_ids'], k=10), axis=1)

# calculate the MAP across the entire query set
print(query_df.loc[:, 'map@k'].mean())